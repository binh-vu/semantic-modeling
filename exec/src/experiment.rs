use mira::prelude::*;
use input::RustInput;
use std::fs::*;
use std::io::*;
use std::path::*;
use csv;
use serde_json;
use prettytable::*;
use mira::settings::conf_learning::*;
use gmtk::prelude::*;
use mira::assembling::learning::gen_train_data::gen_train_data;
use std::collections::HashMap;
use algorithm::data_structure::graph::Graph;
use mira::assembling::predicting::Prediction;

/// Get default model from a given input
fn get_default_model<'a>(input: &'a RustInput) -> MRRModel<'a> {
    let workdir = Path::new(&input.workdir);
    let annotator: Annotator = input.get_annotator();

    let mut train_examples = input.iter_train_sms()
        .map(|sm| annotator.create_labeled_mrr_example(&sm.id, sm.graph.clone()))
        .filter(|e| e.is_some())
        .map(|e| e.unwrap())
        .collect::<Vec<_>>();
    let mut test_examples = input.iter_test_sms()
        .map(|sm| annotator.create_labeled_mrr_example(&sm.id, sm.graph.clone()))
        .filter(|e| e.is_some())
        .map(|e| e.unwrap())
        .collect::<Vec<_>>();
    
    let model_file = workdir.join("model.default.bin");
    let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut test_examples, &Settings::get_instance().mrf);
    mrr_model.serialize(&model_file);
    // also serialize train/test examples
    serde_json::to_writer(File::create(workdir.join("examples.train.default.json")).unwrap(), &train_examples).unwrap();
    serde_json::to_writer(File::create(workdir.join("examples.test.default.json")).unwrap(), &test_examples).unwrap();
    mrr_model
}

pub fn load_model<'a>(input: &'a RustInput, model_file: &str) -> MRRModel<'a> {
    let workdir = Path::new(&input.workdir);
    let annotator: Annotator = input.get_annotator();
    return MRRModel::deserialize(annotator, &workdir.join(model_file));
}

pub fn evaluate(evaluate_sms: &[&SemanticModel], pred_sms: &[Prediction], foutput: &Path) {
    let mut wtr = csv::Writer::from_writer(File::create(foutput).unwrap());
    let mut table = Table::new();
    wtr.write_record(&["source", "precision", "recall", "f1", "stype-acc", "map-precision", "map-recall", "map-f1", "map-stype-acc", "oracle-precision", "oracle-recall", "oracle-f1", "oracle-stype-acc"]).unwrap();
    table.add_row(row!["source", "precision", "recall", "f1", "stype-acc", "map-precision", "map-recall", "map-f1", "map-stype-acc", "oracle-precision", "oracle-recall", "oracle-f1", "oracle-stype-acc"]);

    let mut total_f1 = 0.0;
    let mut total_precision = 0.0;
    let mut total_recall = 0.0;
    let mut total_stype_acc = 0.0;

    let mut total_map_f1 = 0.0;
    let mut total_map_precision = 0.0;
    let mut total_map_recall = 0.0;
    let mut total_map_stype_acc = 0.0;

    let mut total_best_f1 = 0.0;
    let mut total_best_precision = 0.0;
    let mut total_best_recall = 0.0;
    let mut total_best_stype_acc = 0.0;

    for (i, sm) in evaluate_sms.iter().enumerate() {
        let (f1, precision, recall, _1, _2) = smodel_eval::f1_precision_recall(&sm.graph, pred_sms[i].get_prediction(), smodel_eval::DataNodeMode::NoTouch, 500000).unwrap();
        let stype_acc = smodel_eval::stype_acc(&sm.graph, pred_sms[i].get_prediction());

        let map_filtered_pred = pred_sms[i].get_map_filtered_prediction();
        let (map_f1, map_precision, map_recall, _1, _2) = smodel_eval::f1_precision_recall(&sm.graph, &map_filtered_pred, smodel_eval::DataNodeMode::NoTouch, 500000).unwrap();
        let map_stype_acc = smodel_eval::stype_acc(&sm.graph, &map_filtered_pred);

        let best_prediction = pred_sms[i].get_best_prediction_by(|g| smodel_eval::f1_precision_recall(&sm.graph, g, smodel_eval::DataNodeMode::NoTouch, 500000).unwrap().0);
        let (best_f1, best_precision, best_recall, _1, _2) = smodel_eval::f1_precision_recall(&sm.graph, best_prediction, smodel_eval::DataNodeMode::NoTouch, 500000).unwrap();
        let best_stype_acc = smodel_eval::stype_acc(&sm.graph, best_prediction);

        wtr.serialize((&sm.id, precision, recall, f1, stype_acc, map_precision, map_recall, map_f1, map_stype_acc, best_precision, best_recall, best_f1, best_stype_acc)).unwrap();
        table.add_row(row![&sm.id, precision, recall, f1, stype_acc, map_precision, map_recall, map_f1, map_stype_acc, best_precision, best_recall, best_f1, best_stype_acc]);
        total_f1 += f1;
        total_precision += precision;
        total_recall += recall;
        total_stype_acc += stype_acc;

        total_map_f1 += map_f1;
        total_map_precision += map_precision;
        total_map_recall += map_recall;
        total_map_stype_acc += map_stype_acc;

        total_best_f1 += best_f1;
        total_best_precision += best_precision;
        total_best_recall += best_recall;
        total_best_stype_acc += best_stype_acc;
    }
    
    wtr.serialize((
        "average", 
        total_precision / evaluate_sms.len() as f64,
        total_recall / evaluate_sms.len() as f64,
        total_f1 / evaluate_sms.len() as f64,
        total_stype_acc / evaluate_sms.len() as f64,
        total_map_precision / evaluate_sms.len() as f64,
        total_map_recall / evaluate_sms.len() as f64,
        total_map_f1 / evaluate_sms.len() as f64,
        total_map_stype_acc / evaluate_sms.len() as f64,
        total_best_precision / evaluate_sms.len() as f64,
        total_best_recall / evaluate_sms.len() as f64,
        total_best_f1 / evaluate_sms.len() as f64,
        total_best_stype_acc / evaluate_sms.len() as f64
    )).unwrap();
    table.add_row(row![
        "average", 
        total_precision / evaluate_sms.len() as f64,
        total_recall / evaluate_sms.len() as f64,
        total_f1 / evaluate_sms.len() as f64,
        total_stype_acc / evaluate_sms.len() as f64,
        total_map_precision / evaluate_sms.len() as f64,
        total_map_recall / evaluate_sms.len() as f64,
        total_map_f1 / evaluate_sms.len() as f64,
        total_map_stype_acc / evaluate_sms.len() as f64,
        total_best_precision / evaluate_sms.len() as f64,
        total_best_recall / evaluate_sms.len() as f64,
        total_best_f1 / evaluate_sms.len() as f64,
        total_best_stype_acc / evaluate_sms.len() as f64
    ]);
    table.printstd();
}

/// Run gen train
pub fn run_gentrain(input: &RustInput, train_sms: &[String], file_prefix: &str) {
    let workdir = Path::new(&input.workdir);
    let annotator = input.get_annotator();
    let discover_sms = input.get_train_sms().into_iter().filter(|sm| {
        for sid in train_sms.iter() {
            if sid == &sm.id {
                return true;
            }
        }

        return false;
    }).collect::<Vec<_>>();
    let gen_data_method = &Settings::get_instance().learning.gen_data_method;
    let examples = gen_train_data(&discover_sms, annotator, &input.ont_graph, gen_data_method);
    let mut writer = BufWriter::new(File::create(workdir.join(format!("{}.{}.json", file_prefix, gen_data_method.get_name()))).unwrap());

    info!("Generated {} examples", examples.len());
    serde_json::to_writer(writer, &examples).unwrap();
}

/// Run experiment
pub fn run_exp(input: &RustInput, gentest: bool) {
    let workdir = Path::new(&input.workdir);
    let gen_data_method = &Settings::get_instance().learning.gen_data_method;
    let mut timer = Timer::start();

    // print current settings
    info!("Run experiment in with data generating method = {:?}", gen_data_method);

    let evaluate_sms = input.get_test_sms();
    timer.reset();
    let mut annotator: Annotator = input.get_annotator();
    let mut train_examples = gen_train_data(&input.get_train_sms()[..], annotator.clone(), &input.ont_graph, gen_data_method);
    let mut test_examples = Vec::new();
    timer.lap_and_report("Finish generating training examples");

    if gentest {
        info!("Generate testset...");
        if let GenTrainDataMethod::IterativeMRRApproach { n_iter, beam_settings, max_candidates_per_round } = gen_data_method {
            let new_gen_method = GenTrainDataMethod::IterativeMRRApproach {
                n_iter: 1,
                beam_settings: beam_settings.clone(),
                max_candidates_per_round: *max_candidates_per_round
            };
            test_examples = gen_train_data(&evaluate_sms[..], annotator.clone(), &input.ont_graph, &new_gen_method);
        } else {
            test_examples = gen_train_data(&evaluate_sms[..], annotator.clone(), &input.ont_graph, gen_data_method);
        }
    }

    serde_json::to_writer(File::create(&workdir.join(format!("examples.train.{}.json", gen_data_method.get_name()))).unwrap(), &train_examples).unwrap();
    serde_json::to_writer(File::create(&workdir.join(format!("examples.test.{}.json", gen_data_method.get_name()))).unwrap(), &test_examples).unwrap();

    timer.reset();
    let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut test_examples, &Settings::get_instance().mrf);
    timer.lap_and_report("Finish training MRR");
    let pred_sms = predicting(&mrr_model, &evaluate_sms[..]);
    timer.lap_and_report("Finish predicting models");
    mrr_model.serialize(&workdir.join(format!("model.{}.bin", gen_data_method.get_name())));
    // serialize results so we can debug it later
    serde_json::to_writer(File::create(&workdir.join("prediction.json")).unwrap(), &pred_sms).unwrap();

    // evaluate and save the result
    evaluate(&evaluate_sms, &pred_sms, &workdir.join("result.csv"));
}

pub fn run_exp_interactive_modeling(input: &RustInput) {
    let workdir = Path::new(&input.workdir);
    let gen_data_method = &Settings::get_instance().learning.gen_data_method;

    // print current settings
    info!("Run experiment in with data generating method = {:?}", gen_data_method);

    let evaluate_sms = input.get_test_sms();
    let mut annotator: Annotator = input.get_annotator();
    let mut train_examples = gen_train_data(&input.get_train_sms()[..], annotator.clone(), &input.ont_graph, gen_data_method);
    let mut test_examples = Vec::new();

    serde_json::to_writer(File::create(&workdir.join(format!("examples.train.{}.json", gen_data_method.get_name()))).unwrap(), &train_examples).unwrap();
    serde_json::to_writer(File::create(&workdir.join(format!("examples.test.{}.json", gen_data_method.get_name()))).unwrap(), &test_examples).unwrap();

    let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut test_examples, &Settings::get_instance().mrf);

    run_simulated_interactive_modeling(input, &mrr_model);
    mrr_model.serialize(&workdir.join(format!("model.{}.bin", gen_data_method.get_name())));
}

/// Train and output result (it reuse the training code in MRR, but also add the prediction result so we can debug easier)
pub fn run_train(input: &RustInput, train_file: &str, test_file: &str, run_prediction: bool, run_map_prediction: bool) {
    let workdir = Path::new(&input.workdir);
    // print current settings
    info!("Run training with the following settings: {:?}", Settings::get_instance());

    let annotator: Annotator = input.get_annotator();
    let mut train_examples: Vec<MRRExample> = serde_json::from_reader(BufReader::new(File::open(workdir.join(train_file)).unwrap())).unwrap();

    for example in train_examples.iter_mut() {
        example.deserialize();
    }
    let mut test_examples: Vec<MRRExample> = serde_json::from_reader(BufReader::new(File::open(workdir.join(test_file)).unwrap())).unwrap();
    for example in test_examples.iter_mut() {
        example.deserialize();
    }
    
    let model_file = workdir.join("model.debug.bin");
    let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut test_examples, &Settings::get_instance().mrf);
    mrr_model.serialize(&model_file);

    // re-predict
    if run_prediction {
        run_predict(input, &mrr_model);
    }
    if run_map_prediction {
        run_predict_map(input, &mrr_model, &train_examples, &test_examples);
    }
}

pub fn run_simulated_interactive_modeling(input: &RustInput, mrr_model: &MRRModel) {
    let workdir = Path::new(&input.workdir);
    let evaluate_sms = input.get_test_sms();
    let pred_sms = simulated_interactive_modeling(&mrr_model, &evaluate_sms);

    let mut wtr = csv::Writer::from_writer(File::create(&workdir.join("interactive_modeling.csv")).unwrap());
    let mut table = Table::new();
    wtr.write_record(&["source", "n_corrections", "n_selections", "n_selections_as_corrections"]).unwrap();
    table.add_row(row!["source", "n_corrections", "n_selections", "n_selections_as_corrections"]);

    let mut total_correction_cost = 0.0;
    let mut total_selection_cost = 0.0;
    let mut total_selection_as_correction_cost = 0.0;

    for (sm_id, user_actions) in pred_sms.iter() {
        total_correction_cost += user_actions.get_n_corrections() as f64;
        total_selection_cost += user_actions.get_n_selections() as f64;
        total_selection_as_correction_cost += user_actions.get_n_selections_as_corrections() as f64;

        wtr.serialize((sm_id, user_actions.get_n_corrections(), user_actions.get_n_selections(), user_actions.get_n_selections_as_corrections())).unwrap();
        table.add_row(row![sm_id, user_actions.get_n_corrections(), user_actions.get_n_selections(), user_actions.get_n_selections_as_corrections()]);
    }

    wtr.serialize((
        "average",
        total_correction_cost / evaluate_sms.len() as f64,
        total_selection_cost / evaluate_sms.len() as f64,
        total_selection_as_correction_cost / evaluate_sms.len() as f64,
    )).unwrap();
    table.add_row(row![
        "average",
        total_correction_cost / evaluate_sms.len() as f64,
        total_selection_cost / evaluate_sms.len() as f64,
        total_selection_as_correction_cost / evaluate_sms.len() as f64,
    ]);
    table.printstd();
}

pub fn run_predict(input: &RustInput, mrr_model: &MRRModel) {
    let workdir = Path::new(&input.workdir);
    let evaluate_sms = input.get_test_sms();
    let pred_sms = predicting(&mrr_model, &evaluate_sms);
    evaluate(&evaluate_sms, &pred_sms, &workdir.join("result.csv"));
    serde_json::to_writer(File::create(&workdir.join("prediction.json")).unwrap(), &pred_sms).unwrap();
}

pub fn run_predict_map(input: &RustInput, mrr_model: &MRRModel, train_examples: &[MRRExample], test_examples: &[MRRExample]) {
    let workdir = Path::new(&input.workdir);
    // save map of each train/test for colorizing examples
    let train_factorss: Vec<_> = train_examples.iter()
        .map(|e| mrr_model.model.get_factors(e))
        .collect();
    let test_factorss: Vec<_> = test_examples.iter()
        .map(|e| mrr_model.model.get_factors(e))
        .collect();

    let mut train_map_examples = train_examples.iter().enumerate()
        .map(|(i, e)| {
            MAPExample::new(
                &e.variables,
                &train_factorss[i],
                Box::new(BeliefPropagation::new(InferProb::MAP, &e.variables, &train_factorss[i], 120)
                ))
        })
        .collect::<Vec<_>>();
    let mut test_map_examples = test_examples.iter().enumerate()
        .map(|(i, e)| {
            MAPExample::new(
                &e.variables,
                &test_factorss[i],
                Box::new(BeliefPropagation::new(InferProb::MAP, &e.variables, &test_factorss[i], 120)
                ))
        })
        .collect::<Vec<_>>();

    let map_train_labels = train_map_examples.iter_mut()
        .enumerate().map(|(i, map)| {
        let map_ass = map.get_map_assignment();
        train_examples[i].variables
            .iter()
            .map(|var| map_ass[&var.get_id()].idx)
            .collect::<Vec<_>>()
    })
        .collect::<Vec<_>>();;
    let map_test_labels = test_map_examples.iter_mut()
        .enumerate().map(|(i, map)| {
        let map_ass = map.get_map_assignment();
        test_examples[i].variables
            .iter()
            .map(|var| map_ass[&var.get_id()].idx)
            .collect::<Vec<_>>()
    })
        .collect::<Vec<_>>();;

    serde_json::to_writer(File::create(&workdir.join("_train_label.json")).unwrap(), &map_train_labels).unwrap();
    serde_json::to_writer(File::create(&workdir.join("_test_label.json")).unwrap(), &map_test_labels).unwrap();
}

pub fn run_predict_fixed(input: &RustInput, mrr_model: &MRRModel, candidate_sm_file: &Path, output_predict_file: &Path, output_eval_file: &Path) {
    let mut reader = BufReader::new(File::open(candidate_sm_file).unwrap());
    let mut candidate_sms: HashMap<String, Vec<Graph>> = serde_json::from_reader(reader).unwrap();

    let mut predictions = Vec::new();
    for sm in input.get_test_sms() {
        let graphs = candidate_sms.remove(&sm.id).unwrap();

        let mut prediction = Prediction::new(&sm.id);
        let mut graph_w_scores = mrr_model.predict_sm_probs(&sm.id, graphs);
        graph_w_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut search_history = Vec::new();
        let mut search_history_score = Vec::new();
        let mut search_history_map = Vec::new();

        for (graph, score) in graph_w_scores {
            search_history.push(graph);
            search_history_score.push(score);
        }

        let graphs_w_labels = mrr_model.predict_sm_labels(&sm.id, search_history);
        let mut search_history = Vec::new();
        for (graph, lbl) in graphs_w_labels {
            search_history.push(graph);
            search_history_map.push(lbl);
        }

        print!("sm = {}, ", sm.id);
        for i in 0..5.min(search_history_score.len()) {
            let (f1, _3, _4, _1, _2) = smodel_eval::f1_precision_recall(&sm.graph, &search_history[i], smodel_eval::DataNodeMode::NoTouch, 1024).unwrap();
            print!("score={:.5}, f1={:.5} | ", search_history_score[i], f1);
        }
        println!("");

        prediction.search_history.push(search_history);
        prediction.search_history_score.push(search_history_score);
        prediction.search_history_map.push(search_history_map);
        predictions.push(prediction);
    }

    evaluate(&input.get_test_sms(), &predictions, output_eval_file);
    serde_json::to_writer(File::create(output_predict_file).unwrap(), &predictions).unwrap();
}