use assembling::models::annotator::Annotator;
use rdb2rdf::models::semantic_model::SemanticModel;
use rdb2rdf::ontology::ont_graph::OntGraph;
use std::collections::HashMap;
use algorithm::prelude::*;
use std::rc::Rc;
use std::cell::RefCell;
use settings::Settings;
use assembling::learning::elimination::discovery::*;
use assembling::searching::beam_search::*;
use settings::conf_learning::Elimination;
use settings::conf_search::DiscoverMethod;
use assembling::searching::discovery::general_approach::{discover as general_discovery, GeneralDiscoveryArgs, GeneralDiscoveryNode, GraphDiscovery, EmpiricalTripleAdviser, GraphExplorerBuilder};
use assembling::searching::discovery::constraint_space::{discover as int_tree_discovery, IntTreeSearchNodeData, IntTreeSearchArgs, IntTreeSearchNode};
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use assembling::searching::banks::MohsenWeightingSystem;

pub(super) fn custom_general_discovery(search_nodes: &ExploringNodes<GraphDiscovery>, args: &mut GeneralDiscoveryArgs, gold_sm: &Graph,
                                       max_permutation: usize,
                                       train_node_storage: Rc<RefCell<Vec<(String, f64, Graph)>>>) -> Vec<GeneralDiscoveryNode> {
    // explore more nodes, and add those nodes back to the store as training data
    let beam_width = args.beam_width;
    args.beam_width = 5000;
    let next_nodes = general_discovery(search_nodes, args);
    train_node_storage.borrow_mut().extend(next_nodes.iter().map(|node| (node.id.clone(), node.score, node.get_graph().clone())));

    args.beam_width = beam_width;
    next_nodes
}


pub(super) fn custom_constraint_discovery(search_nodes: &ExploringNodes<IntTreeSearchNodeData>, args: &mut IntTreeSearchArgs, gold_sm: &Graph, max_permutation: usize, train_node_storage: Rc<RefCell<Vec<(String, f64, Graph)>>>) -> Vec<IntTreeSearchNode> {
    let beam_width = args.beam_width;
    args.beam_width = 5000;

    let next_nodes = int_tree_discovery(search_nodes, args);
    train_node_storage.borrow_mut().extend(next_nodes.iter().map(|node| (node.id.clone(), node.score, node.get_graph().clone())));

    args.beam_width = beam_width;
    next_nodes
}

pub fn generate_candidate_sms<'a>(annotator: &Annotator<'a>, sm: &SemanticModel, elimination: &Elimination, int_graph: Option<&IntGraph>, ont_graph: Option<&OntGraph>) -> UniqueArray<(Graph, f64)> {
    let settings = Settings::get_instance();
    let search_filter = SearchFilter::new(settings, false);
    let pre_sm_filter = |g: &Graph| search_filter.filter(g);
    let search_node_trackers = Rc::new(RefCell::new(Vec::new()));
    let mut candidate_sms = UniqueArray::<(Graph, f64)>::new();

    let started_nodes = match elimination.n_elimination {
        1 => get_started_eliminated_1nodes(annotator, sm),
        2 => get_started_eliminated_2nodes(annotator, sm),
        _ => unreachable!()
    };

    match elimination.discover_method {
        DiscoverMethod::GeneralDiscovery(ref conf) => {
            let results = {
                let prob_sms = |graphs: Vec<Graph>| {
                    graphs.into_iter().map(|g| (g, 1.0)).collect::<Vec<_>>()
                };
                let discover_func = |search_nodes: &ExploringNodes<GraphDiscovery>, args: &mut GeneralDiscoveryArgs| {
                    custom_general_discovery(search_nodes, args, &sm.graph,
                                             settings.learning.max_permutation,
                                             Rc::clone(&search_node_trackers))
                };

                let mut triple_adviser = EmpiricalTripleAdviser::new(
                    &annotator.train_sms,
                    &sm.attrs,
                    conf.triple_adviser_max_candidate,
                );
                let graph_explorer_builder = GraphExplorerBuilder::new(&mut triple_adviser, conf.max_data_node_hop, conf.max_class_node_hop);

                let mut args = SearchArgs {
                    sm_id: &sm.id,
                    beam_width: conf.beam_width,
                    n_results: conf.beam_width,
                    enable_tracking: false,
                    tracker: Vec::new(),
                    discover: &discover_func,
                    should_stop: &no_stop,
                    extra_args: GeneralDiscoveryArgs {
                        beam_width: conf.beam_width,
                        prob_candidate_sms: &prob_sms,
                        pre_sm_filter: &pre_sm_filter,
                        graph_explorer_builder,
                    },
                    compare_search_node: &default_compare_search_node
                };

                let started_graphs = convert_to_general_discovery_nodes(&sm, started_nodes, &mut args.extra_args);
                beam_search(started_graphs, &mut args)
            };

            for (id, score, graph) in Rc::try_unwrap(search_node_trackers).unwrap().into_inner() {
                candidate_sms.push(id, (graph, score));
            }
            for search_node in results {
                candidate_sms.push(search_node.id, (search_node.data.current_graph, search_node.score));
            }
        },
        DiscoverMethod::ConstraintSpace(ref conf) => {
            let results = {
                let discover_func = |search_nodes: &ExploringNodes<IntTreeSearchNodeData>, args: &mut IntTreeSearchArgs| {
                    custom_constraint_discovery(search_nodes, args, &sm.graph,
                                                settings.learning.max_permutation,
                                                Rc::clone(&search_node_trackers))
                };
                let mut new_int_graph = int_graph.unwrap().adapt_new_source(sm, ont_graph);
                MohsenWeightingSystem::new(&new_int_graph, &annotator.train_sms).weight(&mut new_int_graph);

                let mut args = SearchArgs {
                    sm_id: &sm.id,
                    beam_width: conf.beam_width,
                    n_results: conf.beam_width,
                    enable_tracking: false,
                    tracker: Vec::new(),
                    discover: &discover_func,
                    should_stop: &no_stop,
                    extra_args: IntTreeSearchArgs::default(sm, conf, new_int_graph),
                    compare_search_node: &default_compare_search_node
                };
                let started_states = convert_to_constraint_nodes(&sm, &args.extra_args.int_graph, started_nodes);
                beam_search(started_states, &mut args)
            };

            for (id, score, graph) in Rc::try_unwrap(search_node_trackers).unwrap().into_inner() {
                candidate_sms.push(id, (graph, score));
            }
            for search_node in results {
                candidate_sms.push(search_node.id, (search_node.data.graph, search_node.score));
            }
        }
    }

    candidate_sms
}

#[cfg(test)]
mod tests {
    use assembling::tests::tests::*;
    use super::*;
    use settings::conf_search::ConstraintSpace;
    use utils::*;
    use assembling::searching::banks::MohsenWeightingSystem;

    #[test]
//    #[ignore]
    pub fn test_generate_candidate_sms() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let elimination = Elimination {
            discover_method: DiscoverMethod::ConstraintSpace(ConstraintSpace::default()),
            n_elimination: 2,
            n_candidates: 10,
        };
        let annotator = input.get_annotator();
        let mut int_graph = IntGraph::new(&input.get_train_sms());
        MohsenWeightingSystem::new(&int_graph, &input.get_train_sms()).weight(&mut int_graph);

        let mut timer = Timer::start();
        for sm in input.get_train_sms() {
            timer.lap_and_report(&format!("Generate candidate for source: {}", sm.id));
            let candidate_sms = generate_candidate_sms(&annotator, sm, &elimination, Some(&int_graph), None);
            println!("N candidates: {}", candidate_sms.len());
            assert!(candidate_sms.len() > 0);
            assert!(candidate_sms[0].1 > 0.0);
            break;
        }
    }

    #[test]
    #[ignore]
    pub fn test_generate_candidate_sms_stable() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let elimination = Elimination {
            discover_method: DiscoverMethod::ConstraintSpace(ConstraintSpace::default()),
            n_elimination: 2,
            n_candidates: 10,
        };

        let int_graph = IntGraph::new(&input.get_train_sms());
        let sm = input.get_train_sms()[0];

        let mut candidate_sms = generate_candidate_sms(&input.get_annotator(), sm, &elimination, Some(&int_graph), None)
            .get_value()
            .into_iter()
            .map(|x| x.0)
            .collect::<Vec<_>>();
        candidate_sms.truncate(500);

        let gold_file = format!("resources/assembling/learning/elimination/{}.json", sm.id);
//        serialize_json(&candidate_sms, &gold_file);

        let gold_candidate_sms: Vec<Graph> = deserialize_json(&gold_file);
        assert_eq!(candidate_sms, gold_candidate_sms);
    }
}