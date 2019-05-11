use algorithm::data_structure::graph::Graph;
use evaluation_metrics::semantic_modeling::*;

pub mod max_f1;
pub mod alignment;

pub type Edge2Label = Vec<bool>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MRRLabel {
    pub edge2label: Edge2Label,
    pub bijection: Bijection,
    pub f1: f64,
    pub precision: f64,
    pub recall: f64
}

impl MRRLabel {
    pub fn n_edge_errors(&self) -> i32 {
        let mut result = 0;
        for i in 0..self.edge2label.len() {
            result += !self.edge2label[i] as i32;
        }

        return result;
    }
}

pub fn label(gold_sm: &Graph, pred_sm: &Graph, max_permutation: usize) -> Option<MRRLabel> {
    return max_f1::max_f1(gold_sm, pred_sm, DataNodeMode::NoTouch, max_permutation);
}

pub fn label_no_ambiguous(gold_sm: &Graph, pred_sm: &Graph, max_permutation: usize) -> Option<MRRLabel> {
    return max_f1::max_f1_no_ambiguous(gold_sm, pred_sm, DataNodeMode::NoTouch, max_permutation);
}