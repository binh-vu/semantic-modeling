use serde_json;
use std::path::PathBuf;
use algorithm::data_structure::graph::*;
use mira::evaluation_metrics::semantic_modeling::*;
use std::fs::File;
use std::ffi::OsStr;

#[derive(Deserialize)]
struct TestCase {
    comment: String,
    gold_sm: Graph,
    pred_sm: Graph,
    data_node_mode: DataNodeMode,
    f1_precision_recall: (f64, f64, f64),
    bijection: Bijection
}


#[test]
fn run_test_from_data() {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests/evaluation_metrics/data");

    for entry in dir.read_dir().unwrap() {
        if let Ok(entry) = entry {
            if entry.path().extension().unwrap_or(OsStr::new("")) == "json" {
                let input: TestCase = serde_json::from_reader(File::open(entry.path()).unwrap()).unwrap();
                
                let (f1, precision, recall, bijection, _x_triples) = f1_precision_recall(&input.gold_sm, &input.pred_sm, input.data_node_mode, 10000).unwrap();
                assert_eq!((f1, precision, recall), input.f1_precision_recall);
                assert_eq!(bijection, input.bijection);
            }
        }
    }
}

