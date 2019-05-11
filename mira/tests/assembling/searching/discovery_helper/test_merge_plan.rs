use std::ffi::OsStr;
use std::fs::File;
use algorithm::prelude::*;
use serde_json;
use std::path::PathBuf;
use std::collections::HashSet;
use mira::assembling::searching::discovery::general_approach::*;

//#[test]
fn test_plan4case1() {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests/assembling/searching/discovery_helper/data");

    let tree_a: Graph =serde_json::from_reader(File::open(dir.join("tree_a.json")).unwrap()).unwrap();
    let tree_b: Graph = serde_json::from_reader(File::open(dir.join("tree_b.json")).unwrap()).unwrap();
    let tree_a_search: GraphExplorer = serde_json::from_reader(File::open(dir.join("tree_a_search.json")).unwrap()).unwrap();
    let tree_b_search: GraphExplorer = serde_json::from_reader(File::open(dir.join("tree_b_search.json")).unwrap()).unwrap();
    let gold_plans: Vec<MergePlan> = serde_json::from_reader(File::open(dir.join("plans.json")).unwrap()).unwrap();

    let plans = make_merge_plans4case1(&tree_a, &tree_b, &tree_a_search, &tree_b_search);
    assert_eq!(plans.len(), gold_plans.len());
    let mut compared_plans: HashSet<usize> = Default::default();
    for plan in plans {
        for (i, gold_plan) in gold_plans.iter().enumerate() {
            if compared_plans.contains(&i) {
                continue;
            }

            if plan == *gold_plan {
                compared_plans.insert(i);
                break;
            }
        }
    }
    assert_eq!(compared_plans.len(), gold_plans.len());
}