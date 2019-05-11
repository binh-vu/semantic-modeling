use algorithm::data_structure::graph::*;
use std::collections::HashMap;
use im::vector::Vector as IVector;

use super::internal_structure::*;
use super::find_best_map::*;
use super::dependent_groups::*;
use std::cmp;
use std::collections::HashSet;
use permutohedron::heap_recursive;
use itertools::Itertools;
use fnv::FnvHashSet;

#[derive(PartialEq, Eq, Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataNodeMode {
    // mean we don't touch anything (note that the label of data_node must be unique)
    NoTouch = 0,
    // mean we ignore label of data node (convert it to DATA_NODE, DATA_NODE2 if there are duplication columns)
    IgnoreLabelDataNode = 1,
    // mean we ignore data node
    IgnoreDataNode = 2
}

#[inline]
pub fn prepare_args<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode) -> Vec<PairLabelGroup<'a>> {
    let mut label2nodes: HashMap<&'a str, PairLabelGroup<'a>> = Default::default();
    for v in gold_sm.iter_class_nodes() {
        label2nodes.entry(&v.label)
            .or_insert(PairLabelGroup::new(LabelGroup::new(gold_sm, Vec::new(), data_node_mode), LabelGroup::new(pred_sm, Vec::new(), data_node_mode)))
            .x.push(v);
    }

    for v in pred_sm.iter_class_nodes() {
        label2nodes.entry(&v.label)
            .or_insert(PairLabelGroup::new(LabelGroup::new(gold_sm, Vec::new(), data_node_mode), LabelGroup::new(pred_sm, Vec::new(), data_node_mode)))
            .x_prime.push(v);
    }

    label2nodes.into_iter().map(|entry| entry.1).collect()
}

#[inline]
pub fn get_dependent_groups<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode) -> (Bijection, Vec<DependentGroups<'a>>, Vec<PairLabelGroup<'a>>) {
    let pair_groups = prepare_args(gold_sm, pred_sm, data_node_mode);
    let mut bijection = Bijection::new(gold_sm.n_nodes, pred_sm.n_nodes);
    let mut map_groups = Vec::new();
    let mut independent_groups = Vec::new();

    if data_node_mode == DataNodeMode::NoTouch {
        // explore the fact that labels of data nodes are unique.
        for x in gold_sm.iter_data_nodes() {
            let x_prime = pred_sm.iter_nodes_by_label(&x.label).next();
            match x_prime {
                None => bijection.push_x(x.id, None),
                Some(x_prime) => bijection.push_both(x.id, x_prime.id)
            }
        }
    } else {
        // need to told bijection data nodes have been mapped
        for x in gold_sm.iter_data_nodes() {
            bijection.x2prime[x.id] = DEFAULT_IGNORE_LABEL_DATA_NODE_ID;
        }
        for x_prime in pred_sm.iter_data_nodes() {
            bijection.prime2x[x_prime.id] = DEFAULT_IGNORE_LABEL_DATA_NODE_ID;
        }
    }

    for pair in pair_groups {
        if cmp::max(pair.x.size(), pair.x_prime.size()) == 1 {
            bijection.push(
                if pair.x.size() == 0 { None } else { Some(pair.x.nodes[0].id) },
                if pair.x_prime.size() == 0 { None } else { Some(pair.x_prime.nodes[0].id) }
            );
            independent_groups.push(pair);
        } else if pair.x.size() == 0 {
            for n in &pair.x_prime.nodes {
                bijection.push_x_prime(None, n.id);
            }
            independent_groups.push(pair);
        } else if pair.x_prime.size() == 0 {
            for n in &pair.x.nodes {
                bijection.push_x(n.id, None);
            }
            independent_groups.push(pair);
        } else {
            map_groups.push(pair);
        }
    }

    // try to do greedy match to reduce size of a pair
//    let mut new_independent_map_group_idx = Vec::new();
//
//    for (pair_idx, pair) in map_groups.iter_mut().enumerate() {
//        let mut new_matches = Vec::new();
//        for x_prime in pair.x_prime.nodes.iter() {
//            if x_prime.n_outgoing_edges == 1 && x_prime.n_incoming_edges == 1 {
//                let ce_x_prime = x_prime.first_outgoing_edge(pred_sm).unwrap();
//                let child_x_prime = ce_x_prime.get_target_node(pred_sm);
//                if !child_x_prime.is_data_node() {
//                    continue;
//                }
//
//                let pe_x_prime = x_prime.first_incoming_edge(pred_sm).unwrap();
//                let parent_x_prime = pe_x_prime.get_source_node(pred_sm);
//                if !bijection.is_pred_node_bounded(parent_x_prime.id) {
//                    continue;
//                }
//
//                let child_x_id = bijection.to_x(child_x_prime.id);
//                // get potential x, it must be parent of child_x_id, has same label with x_prime, and has only one child
//                let potential_x = if child_x_id == DEFAULT_IGNORE_LABEL_DATA_NODE_ID {
//                    // iterate through parent_x to find out
//                    let parent_x_id = bijection.to_x(parent_x_prime.id) as usize;
//                    match pair.x.nodes.iter()
//                        .find(|x| {
//                            match x.first_incoming_edge(gold_sm) {
//                                None => false,
//                                Some(ex) => {
//                                    if ex.label != pe_x_prime.label || ex.source_id != parent_x_id || x.n_outgoing_edges != 1 {
//                                        false
//                                    } else {
//                                        x.first_outgoing_edge(gold_sm).unwrap().label == ce_x_prime.label
//                                    }
//                                }
//                            }
//                        }) {
//                        None => {
//                            continue;
//                        },
//                        Some(x) => {
//                            x
//                        }
//                    }
//                } else {
//                    let px = gold_sm.get_node_by_id(child_x_id as usize).first_parent(gold_sm).unwrap();
//                    if px.label != x_prime.label || px.n_outgoing_edges != 1 || px.n_incoming_edges != 1 {
//                        continue;
//                    }
//
//                    px
//                };
//
//                let pe_x = potential_x.first_incoming_edge(gold_sm).unwrap();
//                let ce_x = potential_x.first_outgoing_edge(gold_sm).unwrap();
//
//                if pe_x.label == pe_x_prime.label && ce_x.label == ce_x_prime.label && pe_x.source_id == bijection.to_x(parent_x_prime.id) as usize {
//                    // perfect match
//                    new_matches.push((potential_x.id, x_prime.id));
//                }
//            }
//        }
//
//        if new_matches.len() > 0 {
//            // okay
//            let mut deleted_x_prime_ids = FnvHashSet::default();
//            let mut deleted_x_ids = FnvHashSet::default();
//
//            for (x_id, x_prime_id) in new_matches {
//                bijection.push_both(x_id, x_prime_id);
//                deleted_x_prime_ids.insert(x_prime_id);
//                deleted_x_ids.insert(x_id);
//            }
//
//            pair.x_prime.nodes.retain(|x_prime| !deleted_x_prime_ids.contains(&x_prime.id));
//            pair.x.nodes.retain(|x| !deleted_x_ids.contains(&x.id));
//
//            independent_groups.push(PairLabelGroup {
//                x: LabelGroup::new(gold_sm, deleted_x_ids.iter().map(|&xid| gold_sm.get_node_by_id(xid)).collect::<Vec<_>>(), data_node_mode),
//                x_prime: LabelGroup::new(pred_sm, deleted_x_prime_ids.iter().map(|&xid| pred_sm.get_node_by_id(xid)).collect::<Vec<_>>(), data_node_mode),
//            });
//
//            if pair.x.size() == 0 {
//                for n in &pair.x_prime.nodes {
//                    bijection.push_x_prime(None, n.id);
//                }
//                new_independent_map_group_idx.push(pair_idx);
//            } else if pair.x_prime.size() == 0 {
//                for n in &pair.x.nodes {
//                    bijection.push_x(n.id, None);
//                }
//                new_independent_map_group_idx.push(pair_idx);
//            }
//        }
//    }
//
//    for &pair_idx in new_independent_map_group_idx.iter().rev() {
//        let map_group = map_groups.swap_remove(pair_idx);
//        if map_group.x.size() > 0 || map_group.x_prime.size() > 0 {
//            independent_groups.push(map_group);
//        }
//    }


    let dependent_groups = DependentGroups::split_by_dependency(map_groups, &bijection);
    (bijection, dependent_groups, independent_groups)
}

pub fn f1_precision_recall<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode, max_permutation: usize) -> Option<(f64, f64, f64, Bijection, TripleSet<'a>)> {
    let (mut bijection, dependent_groups, independent_groups) = get_dependent_groups(gold_sm, pred_sm, data_node_mode);

    let n_permutation: usize = dependent_groups.iter().map(|g| g.get_n_permutations()).sum();
    if n_permutation > max_permutation {
        debug!("Number of permutation is too big: {}", n_permutation);
        return None
    }

    for dependent_group in &dependent_groups {
        bijection = find_best_map(dependent_group, bijection);
    }

    let mut all_groups = DependentGroups::combine(dependent_groups);
    all_groups.extends(independent_groups);

    let tp = eval_score(&all_groups, &bijection);

    let recall = tp / cmp::max(all_groups.x_triples.len(), 1) as f64;
    let precision = tp / cmp::max(all_groups.x_prime_triples.len(), 1) as f64;
    let f1 = if tp == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    Some((f1, precision, recall, bijection, all_groups.x_triples))
}

#[cfg(test)]
pub mod tests {
    use serde_json;
    use regex::Regex;
    use std::fs::File;
    use std::path::*;
    use std::io::*;
    use std::ffi::OsStr;
    use std::collections::HashMap;
    use algorithm::prelude::*;
    use evaluation_metrics::semantic_modeling::*;

    #[test]
    pub fn smoke_test() {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        dir.push("resources/evaluation_metrics/semantic_modeling");

        for entry in dir.read_dir().unwrap() {
            if let Ok(entry) = entry {
                if entry.path().extension().unwrap_or(OsStr::new("")) == "json" {
                    let (bijection, fpr, gold_sm, pred_sm) = load_input(entry.path());

                    let (f1, precision, recall, bijection, _x_triples) = f1_precision_recall(&gold_sm, &pred_sm, DataNodeMode::NoTouch, 81000).unwrap();
                    assert!((f1 - fpr.0).abs() <= 1e-8);
                    assert!((precision - fpr.1).abs() <= 1e-8);
                    assert!((recall - fpr.2).abs() <= 1e-8);
                    assert_eq!(bijection, bijection);
                }
            }
        }
    }

    pub fn load_input(test_file: PathBuf) -> (Bijection, (f64, f64, f64), Graph, Graph) {
        let mut input: HashMap<String, serde_json::Value> = serde_json::from_reader(BufReader::new(File::open(test_file).unwrap())).unwrap();
        let gold_graph: Vec<String> = serde_json::from_value(input.remove("gold_sm").unwrap()).unwrap();
        let pred_graph: Vec<String> = serde_json::from_value(input.remove("pred_sm").unwrap()).unwrap();

        let (gold_sm, gold_idmap) = quick_graph(&gold_graph);
        let (pred_sm, pred_idmap) = quick_graph(&pred_graph);

        let f1_precision_recall: (f64, f64, f64) = serde_json::from_value(input.remove("f1_precision_recall").unwrap()).unwrap();
        let raw_bijection: HashMap<String, String> = serde_json::from_value(input.remove("bijection").unwrap()).unwrap();
        let mut bijection = Bijection::new(gold_sm.n_nodes, pred_sm.n_nodes);

        for (x, x_prime) in raw_bijection.iter() {
            let x_id = gold_idmap[x];
            let x_prime_id = pred_idmap[x_prime];

            bijection.push_both(x_id, x_prime_id);
        }

        (bijection, f1_precision_recall, gold_sm, pred_sm)
    }

    pub fn get_nodes<'a>(graph: &'a Graph, nodes: &[usize]) -> Vec<&'a Node> {
        nodes.iter()
            .map(|&nid| graph.get_node_by_id(nid))
            .collect::<Vec<_>>()
    }
}