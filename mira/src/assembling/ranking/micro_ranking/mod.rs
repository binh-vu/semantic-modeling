use settings::conf_ranking::MicroRankingConf;
use std::cmp::Ordering;
use super::ranking_features::*;
use super::Ranking;
use algorithm::prelude::Graph;
use rdb2rdf::prelude::SemanticModel;

pub struct MicroRanking {
    sms: Vec<SemanticModel>,
    coherence: Coherence,
    trigger_delta: f64,
    coherence_weight: f64,
    minimal_weight: f64
}

impl MicroRanking {
    pub fn from_settings(sms: Vec<SemanticModel>, train_sms: &[&SemanticModel], conf: &MicroRankingConf) -> MicroRanking {
        let coherence = Coherence::new(train_sms);

        MicroRanking {
            sms,
            coherence,
            trigger_delta: conf.trigger_delta,
            coherence_weight: conf.coherence_weight,
            minimal_weight: conf.minimal_weight,
        }
    }
}

impl Ranking for MicroRanking {
    fn compare_search_node(&self, sm_idx: usize, a_score: f64, b_score: f64, a_graph: &Graph, b_graph: &Graph) -> Ordering {
        // return reverse coherence so that
        let a_coherence = self.coherence.mohsen_coherence(&a_graph);
        let b_coherence = self.coherence.mohsen_coherence(&b_graph);

        let u = 2 * self.sms[sm_idx].attrs.len();
        let a_size_reduction = (u - a_graph.n_nodes) as f64 / (u - a_graph.get_n_data_nodes()) as f64;
        let b_size_reduction = (u - b_graph.n_nodes) as f64 / (u - b_graph.get_n_data_nodes()) as f64;

        let a_weight = a_coherence * self.coherence_weight + a_size_reduction as f64 * self.minimal_weight;
        let b_weight = b_coherence * self.coherence_weight + b_size_reduction as f64 * self.minimal_weight;

        let ordering = (a_score + a_weight).partial_cmp(&(b_score + b_weight)).unwrap();
//        let ordering = if (a_score - b_score).abs() <= self.trigger_delta {
//            let a_coherence = self.coherence.mohsen_coherence(&a_graph);
//            let b_coherence = self.coherence.mohsen_coherence(&b_graph);
//
//            let u = 2 * self.sms[sm_idx].attrs.len();
//            let a_size_reduction = (u - a_graph.n_nodes) as f64 / (u - a_graph.get_n_data_nodes()) as f64;
//            let b_size_reduction = (u - b_graph.n_nodes) as f64 / (u - b_graph.get_n_data_nodes()) as f64;
//
//            let a_weight = a_coherence * self.coherence_weight + a_size_reduction as f64 * self.minimal_weight;
//            let b_weight = b_coherence * self.coherence_weight + b_size_reduction as f64 * self.minimal_weight;
//
//            if a_weight == b_weight {
//                a_score.partial_cmp(&b_score).unwrap()
//            } else {
//                a_weight.partial_cmp(&b_weight).unwrap()
//            }
//        } else {
//            a_score.partial_cmp(&b_score).unwrap()
//        };

        ordering.reverse()
    }

    fn get_rank_score(&self, sm_idx: usize, score: f64, graph: &Graph) -> f64 {
        let coherence = self.coherence.mohsen_coherence(&graph);

        let u = 2 * self.sms[sm_idx].attrs.len();
        let size_reduction = (u - graph.n_nodes) as f64 / (u - graph.get_n_data_nodes()) as f64;

        coherence * self.coherence_weight + size_reduction as f64 * self.minimal_weight
    }
}