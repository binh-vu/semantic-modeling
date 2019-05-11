use assembling::learning::trial_and_error::discovery::get_started_nodes;
use assembling::models::annotator::Annotator;
use rdb2rdf::models::semantic_model::SemanticModel;
use rdb2rdf::ontology::ont_graph::OntGraph;
use std::collections::HashMap;
use algorithm::data_structure::graph::*;
use assembling::searching::beam_search::*;

use assembling::learning::trial_and_error::data_structure::*;
use assembling::learning::trial_and_error::discovery::discover;
use std::rc::Rc;
use std::cell::RefCell;
use settings::{Settings, conf_search};
use assembling::auto_label;
use std::collections::HashSet;
use algorithm::data_structure::unique_array::UniqueArray;
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use itertools::Itertools;


pub fn recursive_remove(nid: usize, graph: &Graph, delete_node_ids: &mut FnvHashSet<usize>, delete_edge_ids: &FnvHashSet<usize>, new_delete_edges: &mut Vec<usize>) {
    delete_node_ids.insert(nid);
    for e in graph.get_node_by_id(nid).iter_outgoing_edges(&graph) {
        new_delete_edges.push(e.id);
        recursive_remove(e.target_id, graph, delete_node_ids, delete_edge_ids, new_delete_edges);
    }
} 


pub(super) fn rollback_incorrect_label(gold_sm: &Graph, max_permutation: usize, search_node: &TrialErrorSearchNode) -> Option<TrialErrorSearchNode> {
    match auto_label::label_no_ambiguous(gold_sm, search_node.get_graph(), max_permutation) {
        None => None,
        Some(mrr_label) => {
            if mrr_label.precision == 1.0 {
                // all nodes are correct, we don't need to rollback
                Some(search_node.clone())
            } else {
                // need to rollback incorrect nodes
                let n_graph = search_node.get_graph();
                let mut g = Graph::new_like(&n_graph);

                // TODO: improve me, may be we can exploit the fact that last edges always incorrect
                // however, the deleted nodes are not always class nodes
                let mut rewire_idmap: FnvHashMap<usize, usize> = Default::default();
                let mut delete_node_ids: FnvHashSet<usize> = Default::default();
                let mut delete_edge_ids: FnvHashSet<usize> = Default::default();
                let cannot_delete_node = search_node.data.mount.as_ref().unwrap().class_id;

                for (i, label) in mrr_label.edge2label.iter().enumerate() {
                    if !label {
                        let del_e = n_graph.get_edge_by_id(i);
                        delete_edge_ids.insert(i);

                        if del_e.get_target_node(&n_graph).n_incoming_edges == 1 {
                            delete_node_ids.insert(del_e.target_id);
                        }

                        let mut cascade_delete_node = del_e.get_source_node(&n_graph);
                        while cascade_delete_node.n_outgoing_edges == 1 {
                            if cascade_delete_node.id == cannot_delete_node {
                                break;
                            }

                            delete_node_ids.insert(cascade_delete_node.id);

                            // no more outgoing edges, we may want to propagate this to upper level
                            match cascade_delete_node.first_incoming_edge(&n_graph) {
                                None => {
                                    break;
                                },
                                Some(pe) => {
                                    cascade_delete_node = pe.get_source_node(&n_graph);
                                    if cascade_delete_node.n_outgoing_edges == 1 {
                                        if cascade_delete_node.id == cannot_delete_node {
                                            break;
                                        }
                                        delete_edge_ids.insert(pe.id);
                                    }
                                }
                            }
                        }
                    }
                }

                // a node may have all children deleted, but it doesn't capture in the code above
                // we need to loop until no more new delete node, but it may slows...
                loop {
                    let mut new_delete_nodes = Vec::new();
                    for &nid in delete_node_ids.iter() {
                        let node = n_graph.get_node_by_id(nid);
                        match node.first_incoming_edge(&n_graph) {
                            None => (),
                            Some(pe) => {
                                // if its parent edge has not been included
                                if !delete_edge_ids.contains(&pe.id) {
                                    delete_edge_ids.insert(pe.id);
                                }

                                // examine if we can delete the parent node as well
                                if !delete_node_ids.contains(&pe.source_id) && cannot_delete_node != pe.source_id {
                                    let parent_node = pe.get_source_node(&n_graph);
                                    if parent_node.outgoing_edges.iter().all(|eid| delete_edge_ids.contains(eid)) {
                                        new_delete_nodes.push(pe.source_id);
                                    }
                                }
                            }
                        }
                    }

                    if new_delete_nodes.len() == 0 {
                        break;
                    } else {
                        delete_node_ids.extend(new_delete_nodes.into_iter());
                    }
                }

                // an incorrect edge may split the graph into 2 graphs
                // detect this case and remove the second graph as well
                let mut new_delete_edges = Vec::new();
                for &eid in &delete_edge_ids {
                    let e = n_graph.get_edge_by_id(eid);
                    recursive_remove(e.target_id, n_graph, &mut delete_node_ids, &delete_edge_ids, &mut new_delete_edges);
                }
                delete_edge_ids.extend(new_delete_edges.into_iter());

                for n in n_graph.iter_nodes() {
                    if !delete_node_ids.contains(&n.id) {
                        rewire_idmap.insert(n.id, g.add_node(Node::new(n.kind, n.label.clone())));
                    }
                }

                // println!("[DEBUG] rewire_idmap = {:?}", rewire_idmap);
                // // === [DEBUG] DEBUG CODE START HERE ===
                // println!("[DEBUG] at generator.rs");
                // use debug_utils::*;
                // println!("[DEBUG] clear dir"); clear_dir();
                // draw_graphs(&[g.clone(), n_graph.clone()]);
                // println!("[DEBUG] delete_node_ids = {:?}", delete_node_ids);
                // println!("[DEBUG] delete_edge_ids = {:?}", delete_edge_ids);
                // === [DEBUG] DEBUG CODE  END  HERE ===
                
                for e in n_graph.iter_edges() {
                    if !delete_edge_ids.contains(&e.id) {
                        g.add_edge(Edge::new(e.kind, e.label.clone(), rewire_idmap[&e.source_id], rewire_idmap[&e.target_id]));
                    }
                }

                if g.n_edges > 0 {
                    let mut deleted_attrs = delete_node_ids.iter()
                        .filter(|&&nid| n_graph.get_node_by_id(nid).is_data_node())
                        .map(|&nid| &n_graph.get_node_by_id(nid).label);
                    
                    let mut remained_attributes = search_node.data.remained_attributes.update(deleted_attrs.next().expect("Must have at least one because it's incorrect").to_owned());
                    for attr in deleted_attrs {
                        remained_attributes.insert(attr.to_owned());
                    }

                    let new_node = TrialErrorNodeData::new(
                        remained_attributes, search_node.data.mount.clone(), g).into_search_node(search_node.score);
                    Some(new_node)
                } else {
                    None
                }
            }
        }
    }
}


fn control_next_nodes(gold_sm: &Graph, max_permutation: usize, next_nodes: &[TrialErrorSearchNode]) -> Vec<TrialErrorSearchNode> {
    let mut unique_id: HashSet<String> = Default::default();
    let mut results: Vec<TrialErrorSearchNode> = Default::default();

    for next_node in next_nodes {
        match rollback_incorrect_label(gold_sm, max_permutation, next_node) {
            None => {},
            Some(n) => {
                if !unique_id.contains(&n.id) {
                    unique_id.insert(n.id.clone());
                    results.push(n);
                }
            }
        }
    }

    results
}


pub(super) fn custom_discovery<'a>(search_nodes: &ExploringNodes<TrialErrorNodeData>, args: &mut TrialErrorSearchArgs<'a>,
                                   gold_sm: &Graph, max_candidate: usize, max_permutation: usize,
                                   visited_search_nodes: Rc<RefCell<HashSet<String>>>,
                                   train_node_storage: Rc<RefCell<Vec<TrialErrorSearchNode>>>) -> Vec<TrialErrorSearchNode> {
    let mut controlled_next_nodes = Vec::new();
    let mut controlled_next_nodes_id: HashSet<String> = Default::default();
    
    // explore more nodes, and add those nodes back to the store as training data
    let beam_width = args.beam_width;
    args.beam_width = 10000;

    // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] at generator.rs =========================================================================================");
    // use debug_utils::*;
    // draw_graphs(&[]);
    // === [DEBUG] DEBUG CODE  END  HERE ===
    for search_node in search_nodes.iter() {
        let mut next_nodes = discover(&vec![search_node.clone()], args);
        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] at generator.rs");
        // for next_node in next_nodes.iter() {
        //     for n in next_node.get_graph().iter_class_nodes() {
        //         let e0 = next_node.get_graph().get_edge_by_id(n.outgoing_edges[0]);
        //         if n.n_outgoing_edges > 5 && n.iter_outgoing_edges(next_node.get_graph()).all(|e| e.label == e0.label) {
        //             println!("[DEBUG] what is wrong??? what is wrong???");
        //         }
        //     }
        // }
        // === [DEBUG] DEBUG CODE END   HERE ===

        // post processing to make sure we only left the one that are correct, and the rest are discard
        let new_controlled_next_nodes = control_next_nodes(gold_sm, max_permutation, &next_nodes);

        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] next_nodes.len() = {}, new_controlled_next_nodes.len() == {}", next_nodes.len(), new_controlled_next_nodes.len());
        // let graphs = next_nodes.iter().map(|n| n.get_graph().clone()).collect::<Vec<_>>();
        // draw_graphs_same_dir(&graphs);
        // === [DEBUG] DEBUG CODE  END  HERE ===

        // store discovered nodes to storage
        train_node_storage.borrow_mut().append(&mut next_nodes);

        if new_controlled_next_nodes.len() == 0 {
            continue;
        }

        let new_node = if new_controlled_next_nodes.len() == 1 {
            // first is we have only one node, and its id = id of current_search_node
            // mean all mount is wrong, so we keep this one, because we want to move on
            // and if this is a class node, we need to mark it as done, otherwise, we won't
            // never stop
            let mut prev = new_controlled_next_nodes.into_iter().next().unwrap();
            if prev.id == search_node.id {
                // === [DEBUG] DEBUG CODE START HERE ===
                // println!("[DEBUG] prev.data.mount = {:?}", prev.data.mount);
                // === [DEBUG] DEBUG CODE END   HERE ===
                if prev.data.mount.as_ref().unwrap().is_class_mount(&prev.data.current_graph, args) {
                    prev.data.mount.as_mut().unwrap().is_done = true;
                }
            }
            prev
        } else {
            // we have more than one nodes, it means some mounts are correct, we keep the max mount
            // which have max n_nodes
            new_controlled_next_nodes.into_iter()
                .max_by(|x, y| x.get_graph().n_nodes.cmp(&y.get_graph().n_nodes))
                .unwrap()
        };

        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] new_node.mount = {:?}", new_node.data.mount);
        // === [DEBUG] DEBUG CODE END   HERE ===
        
        if !controlled_next_nodes_id.contains(&new_node.id) {
            controlled_next_nodes_id.insert(new_node.id.clone());
            controlled_next_nodes.push(new_node);
        }
    }

    args.beam_width = beam_width;

    // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] at generator.rs");
    // use debug_utils::*;
    // // println!("[DEBUG] clear dir"); clear_dir();
    // let graphs = controlled_next_nodes.iter().map(|n| n.get_graph().clone()).collect::<Vec<_>>();
    // draw_graphs(&graphs);
    // === [DEBUG] DEBUG CODE  END  HERE ===

    controlled_next_nodes
}

/// Generate candidate sms using trial and error methods
pub fn generate_candidate_sms<'a, F>(prob_sms: F, annotator: &Annotator<'a>, sm: &SemanticModel, _ont_graph: &OntGraph, max_candidates_per_round: usize, beam_width: usize) -> HashMap<String, Graph>
    where F: Fn(Vec<Graph>) -> Vec<(Graph, f64)> {
    let settings = Settings::get_instance();
    let search_node_trackers = Rc::new(RefCell::new(Vec::new()));
    let visited_search_nodes = Rc::new(RefCell::new(HashSet::new()));

    // // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] at generator.rs");
    // use debug_utils::*;
    // println!("[DEBUG] clear dir"); clear_dir();
    // // === [DEBUG] DEBUG CODE  END  HERE ===

    let results = {
        let discover_func = |search_nodes: &ExploringNodes<TrialErrorNodeData>, args: &mut TrialErrorSearchArgs| {
            custom_discovery(search_nodes, args, &sm.graph,
                             max_candidates_per_round, settings.learning.max_permutation,
                             Rc::clone(&visited_search_nodes),
                             Rc::clone(&search_node_trackers))
        };

        let mut args = SearchArgs {
            sm_id: &sm.id,
            beam_width,
            n_results: beam_width,
            enable_tracking: false,
            tracker: vec![],
            discover: &discover_func,
            should_stop: &no_stop,
            compare_search_node: &default_compare_search_node,
            extra_args: TrialErrorSearchArgs {
                beam_width,
                prob_candidate_sms: &prob_sms,
                trial_error_exploring:TrialErrorExploring::new(annotator, sm),
                sm,
                max_n_duplications: settings.mrf.max_n_duplications
            },
        };

        let started_nodes = get_started_nodes(annotator, sm, &args.extra_args);
        beam_search(started_nodes, &mut args)
    };

    let mut candidate_sms: HashMap<String, Graph> = Default::default();

    for search_node in Rc::try_unwrap(search_node_trackers).unwrap().into_inner() {
        if search_node.get_graph().n_edges == 0 {
            continue;
        }

        candidate_sms.insert(search_node.id, search_node.data.current_graph);
    }
    for search_node in results {
        if search_node.get_graph().n_edges == 0 {
            continue;
        }

        candidate_sms.insert(search_node.id, search_node.data.current_graph);
    }

    candidate_sms
}