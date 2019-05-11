use algorithm::prelude::*;
use rdb2rdf::prelude::*;
use super::search_node_data::*;
use super::merge_plan::*;
use assembling::searching::banks::*;
use settings::conf_search::{ConstraintSpace, CPMergePlanFilter};

pub type SMFilterFunc<'a> = &'a Fn(&Graph, &IntTreeSearchArgs) -> bool;
pub type MergePlanFilterFunc<'a> = &'a Fn(&Graph, &MergePlan, &IntTreeSearchArgs) -> bool;
pub type ProbCandidateSMFunc<'a> = &'a Fn(Vec<Graph>, &[Bijection], &IntTreeSearchArgs) -> Vec<(Graph, f64)>;

pub struct IntTreeSearchArgs<'a> {
    pub sm: &'a SemanticModel,
    pub int_graph: IntGraph,
    pub path_index: GraphPathIndex,
    pub ancestor_index: AncestorIndex,
    pub sm_filter: SMFilterFunc<'a>,
    pub prob_candidate_sms: ProbCandidateSMFunc<'a>,
    pub merge_plan_filter: MergePlanFilterFunc<'a>,
    pub beam_width: usize,
    pub merge_plan_filter_conf: CPMergePlanFilter
}

impl<'a> IntTreeSearchArgs<'a> {
    pub fn default(sm: &'a SemanticModel, constraint_space_args: &ConstraintSpace, int_graph: IntGraph) -> IntTreeSearchArgs<'a> {
        IntTreeSearchArgs {
            sm,
            beam_width: constraint_space_args.beam_width,
            path_index: GraphPathIndex::new(&int_graph.graph).unwrap(),
            ancestor_index: AncestorIndex::new(&int_graph.graph),
            int_graph,
            sm_filter: &default_sm_filter,
            prob_candidate_sms: &default_prob_sm,
            merge_plan_filter: &default_merge_plan_filter,
            merge_plan_filter_conf: constraint_space_args.merge_plan_filter.clone()
        }
    }

    pub fn default_with_prob_candidate_sms(sm: &'a SemanticModel, constraint_space_args: &ConstraintSpace, prob_candidate_sms: ProbCandidateSMFunc<'a>, int_graph: IntGraph) -> IntTreeSearchArgs<'a> {
        IntTreeSearchArgs {
            sm,
            beam_width: constraint_space_args.beam_width,
            path_index: GraphPathIndex::new(&int_graph.graph).unwrap(),
            ancestor_index: AncestorIndex::new(&int_graph.graph),
            int_graph,
            sm_filter: &default_sm_filter,
            merge_plan_filter: &default_merge_plan_filter,
            prob_candidate_sms,
            merge_plan_filter_conf: constraint_space_args.merge_plan_filter.clone()
        }
    }

    pub fn new(sm: &'a SemanticModel, constraint_space_args: &ConstraintSpace, sm_filter: SMFilterFunc<'a>, merge_plan_filter: MergePlanFilterFunc<'a>, prob_candidate_sms: ProbCandidateSMFunc<'a>, int_graph: IntGraph) -> IntTreeSearchArgs<'a> {
        IntTreeSearchArgs {
            sm,
            beam_width: constraint_space_args.beam_width,
            sm_filter,
            path_index: GraphPathIndex::new(&int_graph.graph).unwrap(),
            ancestor_index: AncestorIndex::new(&int_graph.graph),
            int_graph,
            prob_candidate_sms,
            merge_plan_filter,
            merge_plan_filter_conf: constraint_space_args.merge_plan_filter.clone()
        }
    }

    pub fn update_index(&mut self) {
        self.path_index = GraphPathIndex::new(&self.int_graph.graph).unwrap();
        self.ancestor_index = AncestorIndex::new(&self.int_graph.graph);
    }
}

pub fn default_merge_plan_filter(graph: &Graph, plan: &MergePlan, args: &IntTreeSearchArgs) -> bool {
    if !args.merge_plan_filter_conf.enable {
        // don't ignore anything
        return true;
    }

    plan.is_good_plan(args.merge_plan_filter_conf.max_n_empty_hop)
}

pub fn default_sm_filter(graph: &Graph, args: &IntTreeSearchArgs) -> bool {
    return true;
}

// Compute the score by max_weights - summing all of the weights (because int IntGraph, less weight is better)
pub fn default_prob_sm(graphs: Vec<Graph>, bijections: &[Bijection], args: &IntTreeSearchArgs) -> Vec<(Graph, f64)> {
    let max_score: f64 = args.int_graph.graph.iter_edges().fold(0.0, |acc, x| acc + x.data.weight) as f64;

    return graphs.into_iter().zip(bijections.into_iter())
        .map(|(g, bijection)| {
            let mut score: f64 = 0.0;
            for e_prime in g.iter_edges() {
                let s_prime = e_prime.get_source_node(&g);
                let t_prime_id = bijection.to_x(e_prime.target_id);
                let s = args.int_graph.graph.get_node_by_id(bijection.to_x(s_prime.id));

                for e in s.iter_outgoing_edges(&args.int_graph.graph) {
                    if e.target_id == t_prime_id && e.label == e_prime.label {
                        score += e.data.weight as f64;
                        break;
                    }
                }
            }

            (g, max_score - score)
        })
        .collect::<Vec<_>>();
}