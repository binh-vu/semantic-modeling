use std::ffi::OsStr;
use std::fs::File;
use algorithm::prelude::*;
use serde_json;
use std::path::PathBuf;
use std::collections::HashSet;
use mira::assembling::searching::discovery::general_approach::*;

#[test]
fn test_merge_plans() {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.push("tests/assembling/searching/discovery_helper/data");

    let tree_a: Graph =serde_json::from_reader(File::open(dir.join("tree_a.json")).unwrap()).unwrap();
    let tree_b: Graph = serde_json::from_reader(File::open(dir.join("tree_b.json")).unwrap()).unwrap();
    let plans: Vec<MergePlan> = serde_json::from_reader(File::open(dir.join("plans.json")).unwrap()).unwrap();
    assert!(plans.len() > 0);
    println!("{}", serde_json::to_string_pretty(&plans[0]).unwrap());

    let mut graphs = Vec::new();
    // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] plans.len() = {}", plans.len());
    // === [DEBUG] DEBUG CODE END   HERE ===
    
    for (i, plan) in plans.into_iter().enumerate() {
        let mg = MergeGraph::new(&tree_a, &tree_b,
            plan.int_tree, plan.int_a, plan.int_b);
        graphs.push(mg.proceed_merging());
    }
//     === [DEBUG] DEBUG CODE START HERE ===
//     println!("[DEBUG] at test_merge_plan.rs");
//     use mira::debug_utils::*;
// //        println!("[DEBUG] clear dir"); clear_dir();
//     draw_graphs(&graphs);
//     === [DEBUG] DEBUG CODE  END  HERE ===


//    assert_eq!(plans[0])
}