use assembling::models::mrr::MRRModel;
use rdb2rdf::models::semantic_model::*;
use algorithm::prelude::*;
use rayon::prelude::*;
use settings::{ Settings, conf_search, conf_predicting };
use assembling::searching::*;
use std::cmp::Ordering;
use assembling::searching::beam_search::*;
use assembling::searching::discovery::*;
use assembling::ranking::{Ranking, MicroRanking};
use assembling::learning::elimination::cascade_remove;
use fnv::FnvHashSet;
use settings::conf_search::DiscoverMethod;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use std::collections::HashSet;
use assembling::auto_label;

#[derive(Serialize, Deserialize)]
pub struct Prediction {
    pub sm_id: String,
    pub search_history: Vec<Vec<Graph>>,
    pub search_history_score: Vec<Vec<f64>>,
    pub search_history_map: Vec<Vec<Vec<bool>>>
}

impl Prediction {

    pub fn new(sm_id: &String) -> Prediction {
        Prediction {
            sm_id: sm_id.clone(),
            search_history: Vec::new(),
            search_history_map: Vec::new(),
            search_history_score: Vec::new()
        }
    }

    pub fn get_prediction(&self) -> &Graph {
        &self.search_history.last().unwrap()[0]
    }

    pub fn get_best_prediction_by<F>(&self, max_by_key: F) -> &Graph
        where F: Fn(&Graph) -> f64 {
        self.search_history.last().unwrap().iter().max_by(|&a, &b| max_by_key(a).partial_cmp(&max_by_key(b)).unwrap()).unwrap()
    }

    pub fn get_map_filtered_prediction(&self) -> Graph {
        let edge2label = &self.search_history_map.last().unwrap()[0];
        let mut remove_edge_ids: FnvHashSet<usize> = Default::default();

        let g = self.get_prediction();
        for e in g.iter_edges() {
            if !edge2label[e.id] {
                if self.get_delete_edge_cost(g, &edge2label, e) < 0.0 {
                    // we can delete this
                    remove_edge_ids.insert(e.id);
                }
            }
        }

        let g = cascade_remove(self.get_prediction(), Default::default(), remove_edge_ids);
        return g;
    }

    fn get_delete_edge_cost(&self, g: &Graph, edge2label: &[bool], edge: &Edge) -> f32 {
        // count number of correct links and incorrect links underneath it
        let descendant_edges = get_descendant_edges(g, edge);
        let n_correct = descendant_edges.iter().filter(|&&e| edge2label[e]).count() as f32;
        let n_incorrect = descendant_edges.len() as f32 - n_correct + 1.0; // default edge_id is wrong
        // cost is n_correct - n_incorrect

        n_correct - n_incorrect
    }
}


fn search_nodes_to_prediction<T: SearchNodeExtraData>(model: &MRRModel, sm: &SemanticModel, tracker_nodes: Vec<Vec<SearchNode<T>>>, results: Vec<SearchNode<T>>) -> Prediction {
    let mut prediction = Prediction {
        sm_id: sm.id.clone(),
        search_history: Vec::new(),
        search_history_map: Vec::new(),
        search_history_score: Vec::new()
    };

    for search_nodes in tracker_nodes.into_iter() {
        let mut search_history = Vec::with_capacity(search_nodes.len());
        let mut search_history_label = Vec::with_capacity(search_nodes.len());
        let mut search_history_score = Vec::with_capacity(search_nodes.len());
        let mut graphs = Vec::with_capacity(search_nodes.len());

        for search_node in search_nodes.into_iter() {
            search_history_score.push(search_node.score);
            graphs.push(search_node.remove_graph());
        }

        let graphs_w_labels = model.predict_sm_labels(&sm.id, graphs);
        for (graph, label) in graphs_w_labels {
            search_history_label.push(label);
            search_history.push(graph);
        }

        prediction.search_history.push(search_history);
        prediction.search_history_map.push(search_history_label);
        prediction.search_history_score.push(search_history_score);
    }

    let mut search_history = Vec::with_capacity(results.len());
    let mut search_history_map = Vec::with_capacity(results.len());
    let mut search_history_score = Vec::with_capacity(results.len());
    let mut graphs = Vec::with_capacity(results.len());

    for search_node in results.into_iter() {
        search_history_score.push(search_node.score);
        graphs.push(search_node.remove_graph());
    }

    let graphs_w_labels = model.predict_sm_labels(&sm.id, graphs);
    for (graph, label) in graphs_w_labels {
        search_history_map.push(label);
        search_history.push(graph);
    }

    prediction.search_history.push(search_history);
    prediction.search_history_map.push(search_history_map);
    prediction.search_history_score.push(search_history_score);

    prediction
}


fn search_for_model(model: &MRRModel, sm: &SemanticModel, ranking: Option<&Box<Ranking>>, search_settings: &conf_search::BeamSearchSettings, int_graph: Option<&IntGraph>) -> Prediction {
    let settings = Settings::get_instance();
    let sm_idx = model.annotator.sm_index[&sm.id];
    
    let search_filter = SearchFilter::new(settings, true);
    let pre_sm_filter = |g: &Graph| search_filter.filter(g);

    match search_settings.discovery {
        DiscoverMethod::ConstraintSpace(ref conf) => {
            let prob_sms = |graphs: Vec<Graph>, bijections: &[constraint_space::Bijection], args: &IntTreeSearchArgs| model.predict_sm_probs(&sm.id, graphs);
            let adapted_int_graph = int_graph.unwrap().adapt_new_source(sm, None);
            let rank_obj;
            let custom_ranking;
            let compare_func: CompareSearchNodeFunc<IntTreeSearchNodeData> = match ranking {
                None => &default_compare_search_node,
                Some(ranking) => {
                    rank_obj = ranking;
                    custom_ranking = |a: &SearchNode<IntTreeSearchNodeData>, b: &SearchNode<IntTreeSearchNodeData>| rank_obj.compare_search_node(sm_idx, a.score, b.score, a.get_graph(), b.get_graph());
                    &custom_ranking
                }
            };

            let mut args = SearchArgs {
                sm_id: &sm.id,
                beam_width: search_settings.beam_width,
                n_results: search_settings.n_results,
                enable_tracking: true,
                tracker: Vec::new(),
                discover: &constraint_space::discover,
                should_stop: &no_stop,
                extra_args: IntTreeSearchArgs::default_with_prob_candidate_sms(sm, conf, &prob_sms, adapted_int_graph),
                compare_search_node: compare_func
            };

            let started_nodes = constraint_space::get_started_pk_nodes(sm, &model.annotator, &mut args.extra_args);
            let results = beam_search(started_nodes, &mut args);

            search_nodes_to_prediction(model, sm, args.tracker, results)
        },
        DiscoverMethod::GeneralDiscovery(ref conf) => {
            let prob_candidate_sms = |graphs: Vec<Graph>| model.predict_sm_probs(&sm.id, graphs);
            let rank_obj;
            let custom_ranking;
            let compare_func: CompareSearchNodeFunc<GraphDiscovery> = match ranking {
                None => &default_compare_search_node,
                Some(ranking) => {
                    rank_obj = ranking;
                    custom_ranking = |a: &SearchNode<GraphDiscovery>, b: &SearchNode<GraphDiscovery>| rank_obj.compare_search_node(sm_idx, a.score, b.score, &a.data.current_graph, &b.data.current_graph);
                    &custom_ranking
                }
            };

            let mut triple_adviser = EmpiricalTripleAdviser::new(
                &model.annotator.train_sms,
                &sm.attrs,
                conf.triple_adviser_max_candidate,
            );
            let graph_explorer_builder = GraphExplorerBuilder::new(&mut triple_adviser, conf.max_data_node_hop, conf.max_class_node_hop);

            let mut args = SearchArgs {
                sm_id: &sm.id,
                beam_width: search_settings.beam_width,
                n_results: search_settings.n_results,
                enable_tracking: true,
                tracker: Vec::new(),
                discover: &general_approach::discover,
                should_stop: &no_stop,
                extra_args: GeneralDiscoveryArgs {
                    beam_width: search_settings.beam_width,
                    prob_candidate_sms: &prob_candidate_sms,
                    pre_sm_filter: &pre_sm_filter,
                    graph_explorer_builder,
                },
                compare_search_node: compare_func
            };

            // let started_nodes = general_approach::get_started_nodes(sm, 10, &mut args);
            let started_nodes = general_approach::get_started_pk_nodes(sm, &model.annotator, &mut args.extra_args);
            let results = beam_search(started_nodes, &mut args);

            search_nodes_to_prediction(model, sm, args.tracker, results)
        }
    }
}

pub fn predicting<'a>(model: &MRRModel<'a>, evaluate_sms: &[&SemanticModel]) -> Vec<Prediction> {
    let settings = Settings::get_instance();

    let ranking: Option<Box<Ranking>> = match settings.predicting.post_ranking {
        conf_predicting::PostRanking::NoPostRanking => None,
        conf_predicting::PostRanking::MicroRanking(ref conf) => {
            Some(Box::new(MicroRanking::from_settings(model.annotator.sms.to_vec(), &model.annotator.train_sms, conf)))
        }
    };

    let int_graph = match settings.predicting.search_method {
        conf_predicting::SearchMethod::BeamSearch(ref bs_conf) => {
            match bs_conf.discovery {
                DiscoverMethod::GeneralDiscovery(ref conf) => None,
                DiscoverMethod::ConstraintSpace(ref conf) => Some(IntGraph::new(&model.annotator.train_sms))
            }
        },
        _ => None
    };
    
    let results = evaluate_sms
        .par_iter()
        .map(|sm| {
            match settings.predicting.search_method {
                conf_predicting::SearchMethod::BeamSearch(ref bs_conf) => {
                    search_for_model(model, sm, ranking.as_ref(), bs_conf, int_graph.as_ref())
                }
            }
        })
        .collect::<Vec<_>>();

    return results;
}