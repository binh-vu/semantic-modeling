use algorithm::data_structure::graph::Graph;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use std::collections::HashMap;

pub struct BankCandidateSM<'a> {
    pub graph: Graph,
    pub int_graph: &'a IntGraph,
    pub edge_idmap: Vec<usize>,
    pub mohsen_coherence_score: f32,
}

impl<'a> BankCandidateSM<'a> {

    pub fn new(graph: Graph, edge_idmap: Vec<usize>, int_graph: &'a IntGraph) -> BankCandidateSM {
        // Vec instead of HashSet because one edge cannot have two tags
        let mut tags: HashMap<&str, Vec<usize>> = Default::default();

        for &eid in &edge_idmap {
            let edge = int_graph.graph.get_edge_by_id(eid);
            for tag in edge.data.tags.iter() {
                tags.entry(tag).or_insert(Vec::new()).push(eid);
            }
        }

        let mohsen_coherence_score = tags.values().max_by_key(|v| v.len()).unwrap().len() as f32;

        BankCandidateSM {
            graph,
            int_graph,
            edge_idmap,
            mohsen_coherence_score
        }
    }
}