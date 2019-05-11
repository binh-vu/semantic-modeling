use assembling::searching::banks::data_structure::int_graph::*;
use rdb2rdf::models::semantic_model::SemanticModel;
use algorithm::data_structure::graph::Graph;


pub struct MohsenWeightingSystem {
    w_l: f32,
    w_h: f32,
    n_known_models: f32
}


impl MohsenWeightingSystem {
    pub fn new(int_graph: &IntGraph, train_sms: &[&SemanticModel]) -> MohsenWeightingSystem {
        let w_l = 1.0;
        let n_links = int_graph.graph.iter_edges()
            .filter(|e| e.data.tags.contains(TAG_FROM_NEW_SOURCE) || e.data.tags.contains(TAG_FROM_ONTOLOGY))
            .count();

        let w_h = w_l * n_links as f32;

        MohsenWeightingSystem {
            w_l,
            w_h,
            n_known_models: train_sms.len() as f32
        }
    }

    pub fn weight(&self, g: &mut IntGraph) {
        for e in g.graph.iter_mut_edges() {
            e.data.weight = if e.data.tags.contains(TAG_FROM_NEW_SOURCE) || e.data.tags.contains(TAG_FROM_ONTOLOGY) {
                self.w_h
            } else {
                self.w_l - e.data.tags.len() as f32 / (self.n_known_models + 1.0)
            };
        }
    }
}