use algorithm::data_structure::graph::Graph;
use rdb2rdf::models::semantic_model::SemanticModel;
use settings::Settings;
use assembling::auto_label;
use evaluation_metrics::semantic_modeling::DataNodeMode;
use rayon::prelude::*;

#[derive(Clone)]
pub struct Coherence {
    max_permutation: usize,
    gold_graphs: Vec<Graph>
}

impl Coherence {
    pub fn new(sms: &[&SemanticModel]) -> Coherence {
        Coherence {
            max_permutation: Settings::get_instance().learning.max_permutation,
            gold_graphs: sms.iter().map(|sm| sm.graph.clone()).collect()
        }
    }

    pub fn mohsen_coherence(&self, g: &Graph) -> f64 {
        // return max number of nodes in g belong to a gold semantic model
        self.gold_graphs.par_iter()
            .map(|gold_g| {
                let result = auto_label::max_f1::max_f1(gold_g, g, DataNodeMode::IgnoreLabelDataNode, self.max_permutation);
                match result {
                    None => 0.0,
                    Some(mrr_label) => mrr_label.precision
                }
            })
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .expect("Number of training graphs must be > 0")
    }
}