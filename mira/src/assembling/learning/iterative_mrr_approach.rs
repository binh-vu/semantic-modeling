use algorithm::data_structure::graph::Graph;
use algorithm::random::random_choice::RandomChoice;
use assembling::auto_label;
use assembling::models::annotator::Annotator;
use assembling::models::example::MRRExample;
use assembling::models::mrr::MRRModel;
use assembling::other_models::BayesModel;
use assembling::searching::beam_search::*;
use assembling::searching::discovery::general_approach::*;
use rand::prelude::*;
use rand::StdRng;
use rayon::prelude::*;
use rdb2rdf::models::semantic_model::SemanticModel;
use rdb2rdf::ontology::ont_graph::OntGraph;
use serde_json;
use settings::{ Settings, conf_search };
use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;


pub(super) fn get_default_model<'a>(annotator: Annotator<'a>) -> MRRModel<'a> {
    let mut train_examples = annotator.sms.iter()
        .map(|sm| annotator.create_labeled_mrr_example(&sm.id, sm.graph.clone()))
        .filter(|e| e.is_some())
        .map(|e| e.unwrap())
        .collect::<Vec<_>>();

    let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut vec![], &Settings::get_instance().mrf);
    mrr_model
}


pub(super) fn custom_discovery(search_nodes: &ExploringNodes<GraphDiscovery>, args: &mut GeneralDiscoveryArgs, gold_sm: &Graph,
                               max_candidates: usize, max_permutation: usize,
                               random_choice: Rc<RefCell<RandomChoice<StdRng>>>,
                               train_node_storage: Rc<RefCell<Vec<GeneralDiscoveryNode>>>) -> Vec<GeneralDiscoveryNode> {
    let max_tolerant_errors = 3;

    // explore more nodes, and add those nodes back to the store as training data
    let beam_width = args.beam_width;
    args.beam_width = 10000;
    let next_nodes = discover(search_nodes, args);
    args.beam_width = beam_width;

    // do a filter out here, if the next nodes have more than 3 errors, stop it
    // also do an oracle selection to make sure it doesn't go wrong!
    let mut controlled_next_nodes = Vec::new();
    for n in next_nodes {
        let mrr_label = auto_label::label_no_ambiguous(gold_sm, n.get_graph(), max_permutation);
        if mrr_label.is_none() || mrr_label.unwrap().n_edge_errors() >= max_tolerant_errors {
            // ignore this examples either there are more than one possible mapping, or contain so many errors
            continue;
        }

        controlled_next_nodes.push(n);
    }

    if controlled_next_nodes.len() == 0 {
        debug!("No valid next states");
        return controlled_next_nodes;
    }

    let next_nodes = (0..cmp::min(args.beam_width, controlled_next_nodes.len()))
        .map(|i| controlled_next_nodes[i].clone())
        .collect::<Vec<_>>();
    // from here you have lots of options to select which examples you want to incorporate into
    // training set.
    //      + 1st: still need to have some correct examples + wild-structures
    //          train_nodes = controlled_next_nodes[:15] + args._tmp_random_state.choice(controlled_next_nodes[15:], size=max_candidates, replace=False)
    //      + 2nd: sampling based on the predicted scores
    //      + 3rd: mixed, currently use this approach
    let mut train_nodes = Vec::new();
    if controlled_next_nodes.len() < max_candidates {
        train_nodes.append(&mut controlled_next_nodes);
    } else {
        let n_priority_boarding = 10;
        let mut economy_class = controlled_next_nodes.split_off(n_priority_boarding);
        let mut economy_index = (0..economy_class.len()).collect::<Vec<_>>();
        let p = economy_class.iter().map(|node| node.score).collect::<Vec<_>>();
        let choices = random_choice.borrow_mut().random_choice_f64(&mut economy_index, &p, max_candidates - n_priority_boarding);

        train_nodes.append(&mut controlled_next_nodes);
        for &i in choices.iter().rev() {
            train_nodes.push(economy_class.swap_remove(*i));
        }
    };

    // store it
    train_node_storage.borrow_mut().append(&mut train_nodes);
    next_nodes
}

pub(super) fn generate_candidate_sms<'a, F>(prob_sms: F, annotator: &Annotator<'a>, sm: &SemanticModel, _ont_graph: &OntGraph, max_candidates_per_round: usize, beam_conf: &conf_search::BeamSearchSettings) -> HashMap<String, Graph>
    where F: Fn(Vec<Graph>) -> Vec<(Graph, f64)> {
    let settings = Settings::get_instance();
    let search_filter = SearchFilter::new(settings, false);
    let pre_sm_filter = |g: &Graph| search_filter.filter(g);
    let search_node_trackers = Rc::new(RefCell::new(Vec::new()));

    let results = {
        let rng = StdRng::from_seed([settings.manual_seed; 32]);
        let mut random_choice = Rc::new(RefCell::new(RandomChoice::new(rng)));

        let discover_func = |search_nodes: &ExploringNodes<GraphDiscovery>, args: &mut GeneralDiscoveryArgs| {
            custom_discovery(search_nodes, args, &sm.graph,
                             max_candidates_per_round, settings.learning.max_permutation,
                             Rc::clone(&random_choice), Rc::clone(&search_node_trackers))
        };

        let general_discovery = beam_conf.discovery.as_general_discovery();
        let mut triple_adviser = EmpiricalTripleAdviser::new(
            &annotator.train_sms,
            &sm.attrs,
            general_discovery.triple_adviser_max_candidate,
        );
        let graph_explorer_builder = GraphExplorerBuilder::new(&mut triple_adviser, general_discovery.max_data_node_hop, general_discovery.max_class_node_hop);

        let mut args = SearchArgs {
            sm_id: &sm.id,
            beam_width: beam_conf.beam_width,
            n_results: beam_conf.n_results,
            enable_tracking: false,
            tracker: Vec::new(),
            discover: &discover_func,
            should_stop: &no_stop,
            extra_args: GeneralDiscoveryArgs {
                beam_width: beam_conf.beam_width,
                prob_candidate_sms: &prob_sms,
                pre_sm_filter: &pre_sm_filter,
                graph_explorer_builder,
            },
            compare_search_node: &default_compare_search_node
        };

        let started_nodes = get_started_nodes(sm, 10, &mut args.extra_args);
        beam_search(started_nodes, &mut args)
    };
    let mut candidate_sms: HashMap<String, Graph> = Default::default();

    for search_node in Rc::try_unwrap(search_node_trackers).unwrap().into_inner() {
        candidate_sms.insert(search_node.id, search_node.data.current_graph);
    }
    for search_node in results {
        candidate_sms.insert(search_node.id, search_node.data.current_graph);
    }

    candidate_sms
}