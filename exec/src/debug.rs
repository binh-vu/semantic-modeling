use input::RustInput;
use mira::assembling::searching::banks::*;
use mira::assembling::searching::banks;
use mira::assembling::predicting::Prediction;
use super::experiment;
use std::path::Path;
use algorithm::prelude::*;
use rdb2rdf::models::semantic_model::SemanticType;
use im::HashMap as IHashMap;
use mira::assembling::searching::banks::generate_candidate_attr_mapping::MappingCandidate;
use experiment::load_model;
use mira::assembling::searching::banks::attributes_mapping::mrr::learn_mrr::get_mrr_train_data;
use mira::assembling::models::mrr::MRRModel;
use mira::settings::Settings;
use std::collections::hash_map::DefaultHasher;
use std::hash::*;

pub fn run(input: &RustInput) {
    let annotator = input.get_annotator();
    println!("Statistic: {:#?}", annotator.statistic);
}

pub fn exp_candidate_attr_mapping(input: &mut RustInput) {
    let workdir = Path::new(&input.workdir);
    let mut annotator = input.get_annotator();
    let train_sms = input.get_train_sms();
    let test_sms = input.get_test_sms();

    let int_graph = IntGraph::new(&train_sms);

     let (mut train_examples, mut test_examples) = get_mrr_train_data(&annotator, &int_graph, &train_sms, &test_sms);
//     let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut test_examples, &Settings::get_instance().mrf);
//     mrr_model.serialize(&workdir.join("model.candidate_mapping.bin"));

//    let mapping_model = learned_mapping_score::LinearRegressionModel::new(&int_graph, &train_sms, &test_sms);
    // let mapping_model = learned_mapping_score::LogisticRegressionModel::new(&int_graph, &train_sms, &test_sms);

    let mut total_f1 = 0.0;
    for sm in &test_sms {
//        let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mapping_model.mapping_score(int_graph, attr_mappings);
         let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mapping_score::mohsen_mapping_score(int_graph, attr_mappings);
//        let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mrr_mapping_score::mrr_mapping_score(&mrr_model, sm, &int_graph, attr_mappings);

        let candidate_attr_mappings = generate_candidate_attr_mapping::generate_candidate_attr_mapping(&int_graph, &sm.attrs, 50, &mut func);
        info!("#candidates mappings: {}", candidate_attr_mappings.len());

        let best_candidates = candidate_attr_mappings.iter()
            .map(|m| eval_mapping_score::evaluate_attr_mapping(&int_graph, sm, m))
            .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap()).unwrap();

        info!("sm: {}, BEST CANDIDATE: {}", sm.id, best_candidates.f1);
        total_f1 += best_candidates.f1;
        // println!("best candidate: {:?}", best_candidates);
    }

    info!("=====================================================> Summary: algorithm: {}", total_f1 / test_sms.len() as f32);
}

pub fn bank_method(input: &mut RustInput, model_file: &str) {
//    exp_candidate_attr_mapping(input);

    let train_sms = input.get_train_sms();
    let test_sms = input.get_test_sms();

    let int_graph = IntGraph::new(&train_sms);

    // === [DEBUG] DEBUG CODE START HERE ===
    println!("[DEBUG] at debug.rs");
    use mira::debug_utils::*;
    println!("[DEBUG] clear dir"); clear_dir();
    draw_graphs(&[int_graph.to_normal_graph()]);
    // === [DEBUG] DEBUG CODE  END  HERE ===

    let workdir = Path::new(&input.workdir);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // build model to rank attribute first
//    let mapping_model = learned_mapping_score::LinearRegressionModel::new(&int_graph, &train_sms, &test_sms);
//    let mapping_model = learned_mapping_score::LogisticRegressionModel::new(&int_graph, &train_sms, &test_sms);

    let mrr_model = load_model(input, model_file);

    // Evaluate attr mappings
//    let mut total_f1 = 0.0;
//    let mut best_total_f1 = 0.0;
//
//    for sm in &test_sms {
////        let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mapping_model.mapping_score(int_graph, attr_mappings);
//        let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mapping_score::mohsen_mapping_score(int_graph, attr_mappings);
////        let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mrr_mapping_score::mrr_mapping_score(&mrr_model, sm, &int_graph, attr_mappings);
//
//        let candidate_attr_mappings = generate_candidate_attr_mapping::generate_candidate_attr_mapping(&int_graph, &sm.attrs, 50, &mut func);
//        info!("#candidates mappings: {}", candidate_attr_mappings.len());
//
////        let oracle_candidates = learned_mapping_score::compute_gold_mapping(&int_graph, sm);
//
//        let best_candidates = candidate_attr_mappings.iter()
//            .map(|m| eval_mapping_score::evaluate_attr_mapping(&int_graph, sm, m))
//            .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap()).unwrap();
//
////        let oracle_eval = eval_mapping_score::evaluate_attr_mapping(&int_graph, sm, &oracle_candidates);
////        info!("BEST CANDIDATE: {}. Upper bound: {}", best_candidates.f1, oracle_eval.f1);
//        info!("sm: {}, BEST CANDIDATE: {}", sm.id, best_candidates.f1);
//        total_f1 += best_candidates.f1;
//        println!("best candidate: {:?}", best_candidates);
//    }
//
//    info!("Summary: algorithm: {}, upperbound: {}", total_f1 / test_sms.len() as f32, best_total_f1);

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // generate candidate semantic models
    let mut predictions = Vec::new();
    for sm in &test_sms {
        let mut func = |int_graph: &IntGraph, attr_mappings: &[MappingCandidate]| mapping_score::mohsen_mapping_score(int_graph, attr_mappings);
        let mut weighted_graph = int_graph.adapt_new_source(sm, None);
        MohsenWeightingSystem::new(&weighted_graph, &train_sms).weight(&mut weighted_graph);

        // let candidate_sms = banks::generate_candidate_sms(&int_graph, sm, &input.ont_graph);
        let candidate_sms = banks::generate_candidate_sms(&weighted_graph, sm, &mut func);
        assert!(candidate_sms.len() > 0);

        let mut pred = Prediction::new(&sm.id);

        let scores = candidate_sms.iter().map(|c| c.mohsen_coherence_score as f64).collect::<Vec<_>>();
        let maps = candidate_sms.iter().map(|c| vec![true; c.graph.n_edges]).collect::<Vec<_>>();
        let graphs = candidate_sms.into_iter().map(|c| c.graph).collect::<Vec<_>>();

        let mut graph_w_scores = mrr_model.predict_sm_probs(&sm.id, graphs);
        graph_w_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let scores = graph_w_scores.iter().map(|gc| gc.1).collect::<Vec<_>>();
        let graphs = graph_w_scores.into_iter().map(|gc| gc.0).collect::<Vec<_>>();
        let graph_w_maps = mrr_model.predict_sm_labels(&sm.id, graphs);
        let maps = graph_w_maps.iter().map(|gc| gc.1.clone()).collect::<Vec<_>>();
        let graphs = graph_w_maps.into_iter().map(|gc| gc.0).collect::<Vec<_>>();

        pred.search_history_score.push(scores);
        pred.search_history.push(graphs);
        pred.search_history_map.push(maps);

        predictions.push(pred);
    }

    experiment::evaluate(&input.get_test_sms(), &predictions, &workdir.join("bank.result.csv"));
}