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
use mira::assembling::predicting::Prediction;
use mira::assembling::ranking::*;
use mira::settings::conf_predicting;
use mira::assembling::searching::discovery::general_approach::GraphDiscovery;

pub fn run_rerank(input: &RustInput, predict_file: &Path) {
    let settings = Settings::get_instance();
    let annotator = input.get_annotator();

    let ranking: Box<Ranking> = match settings.predicting.post_ranking {
        conf_predicting::PostRanking::NoPostRanking => {
            Box::new(DefaultRanking {})
        },
        conf_predicting::PostRanking::MicroRanking(ref conf) => {
            Box::new(MicroRanking::from_settings(annotator.sms.to_vec(), &annotator.train_sms, conf))
        }
    };

    let predictions: Vec<Prediction> = serde_json::from_reader(File::open(predict_file).unwrap()).unwrap();
    for (i, sm) in input.get_test_sms().iter().enumerate() {
        for (iter_no, graphs) in predictions[i].search_history.iter().enumerate() {
            let mut graphs_w_index = graphs.into_iter().enumerate().collect::<Vec<_>>();
            let g_scores = &predictions[i].search_history_score[iter_no];
            let sm_idx = annotator.sm_index[&sm.id];

            graphs_w_index.sort_by(|a, b| {
                ranking.compare_search_node(sm_idx, g_scores[a.0], g_scores[b.0], &a.1, &b.1)
            });

            for &(idx, graph) in graphs_w_index.iter() {
                let (f1, precision, recall, _1, _2) = smodel_eval::f1_precision_recall(&sm.graph, graph, smodel_eval::DataNodeMode::NoTouch, 10240).unwrap();
                println!("sm: {} -- iter: {} -- idx: {} -- score: {} -- rank_score: {} -- precision: {} -- recall: {} -- f1: {}", sm.id, iter_no, idx, g_scores[idx], ranking.get_rank_score(sm_idx, g_scores[idx], graph), precision, recall, f1);
            }
        }
    }
}