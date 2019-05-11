mod ranking_features;
mod micro_ranking;

use std::cmp::Ordering;
use algorithm::prelude::Graph;
pub use self::micro_ranking::MicroRanking;

pub trait Ranking: Sync {
    fn compare_search_node(&self,sm_idx: usize, a_score: f64, b_score: f64, a_graph: &Graph, b_graph: &Graph) -> Ordering;

    fn get_rank_score(&self, sm_idx: usize, score: f64, graph: &Graph) -> f64;
}

pub struct DefaultRanking {}

impl Ranking for DefaultRanking {
    fn compare_search_node(&self, sm_idx: usize, a_score: f64, b_score: f64, a_graph: &Graph, b_graph: &Graph) -> Ordering {
        b_score.partial_cmp(&a_score).unwrap()
    }

    fn get_rank_score(&self, sm_idx: usize, score: f64, graph: &Graph) -> f64 {
        score
    }
}