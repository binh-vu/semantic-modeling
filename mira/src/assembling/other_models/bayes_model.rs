use utils::dict_get;
use rayon::prelude::*;
use assembling::features::statistic::Statistic;
use assembling::models::annotator::Annotator;
use std::collections::HashMap;
use algorithm::data_structure::graph::*;

pub struct BayesModel<'a> {
    stat: &'a Statistic,
    sm_index: &'a HashMap<String, usize>,
    stype_scores: Vec<HashMap<String, HashMap<(String, String), f32>>>,
    default_prior: f32
}

impl<'a> BayesModel<'a> {

    pub fn new(annotator: &'a Annotator<'a>) -> BayesModel<'a> {
        let stype_scores = annotator.sms.iter()
            .map(|sm| {
                let mut scores: HashMap<String, HashMap<(String, String), f32>> = Default::default();
                for attr in &sm.attrs {
                    let mut score: HashMap<(String, String), f32> = Default::default();
                    for stype in &attr.semantic_types {
                        score.insert((stype.class_uri.clone(), stype.predicate.clone()), stype.score);
                    }
                    scores.insert(attr.label.clone(), score);
                }

                scores
            })
            .collect();

        BayesModel {
            stat: &annotator.statistic,
            sm_index: &annotator.sm_index,
            stype_scores,
            default_prior: 0.1
        }
    }

    pub fn predict_sm_probs(&self, sm_id: &str, graphs: Vec<Graph>) -> Vec<(Graph, f64)> {
        let sm_idx = self.sm_index[sm_id];

        graphs.into_par_iter()
            .map(|g| {
                let mut log_prob = 0.0;
                for r in g.iter_nodes() {
                    if r.n_incoming_edges == 0 {
                        log_prob += self.stat.p_n(&r.label, self.default_prior);
                        self.log_prob_tree_given_node(sm_idx, &r, &g);
                    }
                }

                (g, log_prob.exp() as f64)
            })
            .collect::<Vec<_>>()
    }

    fn log_prob_tree_given_node(&self, sm_idx: usize, node: &Node, graph: &Graph) -> f32 {
        let mut log_prob = 0.0;

        for e in node.iter_outgoing_edges(graph) {
            let target = e.get_target_node(graph);
            if target.is_data_node() {
                log_prob += dict_get(&self.stype_scores[sm_idx][&target.label], &node.label, &e.label).unwrap();
            } else {
                log_prob += (self.stat.p_l_given_s(&node.label, &e.label, self.default_prior) + self.stat.p_o_given_sl(&node.label, &e.label, &target.label, self.default_prior)).ln();
                log_prob += self.log_prob_tree_given_node(sm_idx, target, graph);
            }
        }

        log_prob
    }
}