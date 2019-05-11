use rdb2rdf::models::semantic_model::SemanticModel;
use std::collections::HashMap;
use std::ops::AddAssign;
use algorithm::data_structure::graph::{Edge, EdgeData};
use fnv::FnvHashMap;

#[derive(Serialize, Deserialize)]
pub struct StatsPredicate {
    p_l_multival: HashMap<String, f32>
}

impl StatsPredicate {
    pub fn new(train_sms: &[&SemanticModel]) -> StatsPredicate {
        // define extra features after this
        let mut p_l_multival: HashMap<String, f32> = Default::default();
        let mut link_usages: HashMap<&str, Vec<i32>> = Default::default();

        for sm in train_sms {
            let g = &sm.graph;
            for n in g.iter_nodes() {
                let edge_count = StatsPredicate::count_edge_label(n.iter_outgoing_edges(g));

                for (lbl, count) in edge_count {
                    link_usages.entry(lbl).or_insert(Vec::new()).push((count == 1) as i32);
                }
            }
        }

        for (lbl, examples) in link_usages {
            let sum: i32 = examples.iter().sum();
            p_l_multival.insert(lbl.to_owned(), sum as f32 / examples.len() as f32);
        }

        StatsPredicate { p_l_multival }
    }

    pub fn count_edge_label<'a, D: 'a + EdgeData, I: Iterator<Item=&'a Edge<D>>>(edges: I) -> HashMap<&'a str, usize> {
        let mut edge_count: HashMap<&'a str, usize> = Default::default();
        for e in edges {
            *edge_count.entry(&e.label).or_insert(0) += 1;
        }
        edge_count
    }

    /// numbering edge labels, start from 1
    pub fn numbering_edge_labels<'a, D: 'a + EdgeData, I: Iterator<Item=&'a Edge<D>>>(edges: I) -> FnvHashMap<usize, usize> {
        let mut accum_numbered_links: HashMap<&String, usize> = Default::default();
        let mut numbered_edges: FnvHashMap<usize, usize> = Default::default();

        for e in edges {
            accum_numbered_links.entry(&e.label).or_insert(0).add_assign(1);
            numbered_edges.insert(e.id, accum_numbered_links[&e.label]);
        }

        numbered_edges
    }

    pub fn prob_multi_val(&self, predicate: &str, count: usize) -> Option<f32> {
        if !self.p_l_multival.contains_key(predicate) || count == 1 {
            None
        } else {
            Some(1.0 - self.p_l_multival[predicate])
        }
    }

    pub fn is_multi_val(&self, predicate: &str) -> Option<bool> {
        if !self.p_l_multival.contains_key(predicate) {
            None
        } else {
            Some(self.p_l_multival[predicate] > 0.01)
        }
    }
}

