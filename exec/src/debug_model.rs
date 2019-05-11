#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::str::FromStr;
use mira::prelude::*;
use gmtk::prelude::*;
use input::RustInput;
use std::fs::*;
use std::path::*;
use std::io::*;
use serde_json;
use fnv::FnvHashMap;
use prettytable::*;
use std::rc::Rc;
use algorithm::prelude::*;
use itertools::Itertools;
use std::collections::HashMap;

pub fn debug_model_provided_graph(input: &RustInput, model_file: &Path, sm_id: &str, graph: &str) {
    // create graph first
    let mrr_model = MRRModel::deserialize(input.get_annotator(), model_file);
    // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] at debug_model.rs");
    // for param in mrr_model.model.get_parameters() {
    //     println!("[DEBUG] param.get_value().to_1darray() = {:?}", param.get_value().to_1darray());
    // }
    // println!("[DEBUG] mrr_model.pk_pairwise_domain = {:?}", mrr_model.pk_pairwise_domain);
    // === [DEBUG] DEBUG CODE END   HERE ===
    
    let sm = mrr_model.annotator.get_sm(sm_id);

    let mut g = Graph::new(sm_id.to_owned(), true, true, true);
    let mut id_map: HashMap<&str, usize> = Default::default();

    for row in graph.split("\n") {
        let (source_node_id, p, target_node_id) = row.split("---").next_tuple().unwrap();
        // id is always constructed by appending 1 number into the end
        let source_node_lbl = &source_node_id[0..source_node_id.len() - 1];
        
        if !id_map.contains_key(source_node_id) {
            id_map.insert(source_node_id, g.add_node(Node::new(NodeType::ClassNode, source_node_lbl.to_owned())));
        }

        match sm.graph.iter_nodes_by_label(&target_node_id).next() {
            None => {
                let target_node_lbl = &target_node_id[0..target_node_id.len() - 1];
                if !id_map.contains_key(target_node_id) {
                    id_map.insert(&target_node_id, g.add_node(Node::new(NodeType::ClassNode, target_node_lbl.to_owned())));
                }
            },
            Some(n) => {
                assert!(n.is_data_node());
                assert!(!id_map.contains_key(&target_node_id));
                id_map.insert(&target_node_id, g.add_node(Node::new(NodeType::DataNode, target_node_id.to_owned())));
            }
        }

        g.add_edge(Edge::new(EdgeType::Unspecified, p.to_owned(), id_map[source_node_id], id_map[target_node_id]));
    }

    // make example from graph
    let mut example = mrr_model.annotator.create_labeled_mrr_example(&sm_id, g).unwrap();
    print_all(&mrr_model, &mut example);
}

pub fn debug_model(input: &RustInput, mrr_model: &MRRModel, train_file: &Path, test_file: &Path, predict_file: &Path, example_ident: &str) {
    let mut train_examples: Vec<MRRExample> = serde_json::from_reader(BufReader::new(File::open(train_file).unwrap())).unwrap();
    let mut test_examples: Vec<MRRExample> = serde_json::from_reader(BufReader::new(File::open(test_file).unwrap())).unwrap();
    let mut predictions: Vec<Prediction> = serde_json::from_reader(BufReader::new(File::open(predict_file).unwrap())).unwrap();

    let mut example = if example_ident.starts_with("train:") {
        let mut e = train_examples.remove(usize::from_str(&example_ident[6..]).unwrap());
        e.deserialize();
        e
    } else if example_ident.starts_with("test:") {
        let mut e = test_examples.remove(usize::from_str(&example_ident[5..]).unwrap());
        e.deserialize();
        e
    } else if example_ident.starts_with("pred:") {
        // pred:s03:<iter_no>:<i>
        let sm_prefix = &example_ident[5..8];
        let last_delim = example_ident.rfind(':').unwrap();
        let iter_no = usize::from_str(&example_ident[9..last_delim]).unwrap();
        let i = usize::from_str(&example_ident[last_delim+1..]).unwrap();

        let mut idx = predictions.len() + 10;
        for (j, pred) in predictions.iter().enumerate() {
            if pred.sm_id.starts_with(sm_prefix) {
                idx = j;
            }
        }
        
        let mut pred = predictions.swap_remove(idx);
        let g = pred.search_history.swap_remove(iter_no).swap_remove(i);
        mrr_model.annotator.create_labeled_mrr_example(&pred.sm_id, g).unwrap()
    } else {
        panic!("Invalid example_ident: {}", example_ident)
    };

    print_all(&mrr_model, &mut example);
}

fn write_factor_graph(model: &MRRModel, example: &mut MRRExample) {
    let factors = model.model.get_factors(example);

    let content = json!({
        "variables": example.variables.iter()
            .map(|v| v.get_label_value().idx)
            .collect::<Vec<_>>(),
        "factors": factors.iter()
            .map(|f| f.get_scores_tensor().view1().to_1darray())
            .collect::<Vec<_>>(),
        "factor_variables": factors.iter()
            .map(|f| f.get_variables().iter().map(|v| v.get_id()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    });

    serde_json::to_writer_pretty(File::create("/tmp/factor_graphs.json").unwrap(), &content).unwrap();
}

fn print_all(model: &MRRModel, example: &mut MRRExample) {
    model.annotator.annotate(example, &model.tf_domain);
    let rc_example = Rc::new(example.clone());
    let rc_tfdomain = Rc::new(model.tf_domain.clone());
    let rc_pk_pairwise_domain = Rc::new(model.pk_pairwise_domain.clone());
    let rc_dup_pairwise_domain = Rc::new(model.dup_pairwise_domain.clone());
    let rc_cooccur_domain = Rc::new(model.cooccur_domain.clone());

    let target_assignment = example.variables.iter()
        .map(|v| (v.get_id(), v.get_domain().get_value(1)))
        .collect::<FnvHashMap<_, _>>();
    let rc_target_assignment = Rc::new(target_assignment.clone());
    let factors = model.model.get_factors(example);

    let mut inference = BeliefPropagation::new(InferProb::MARGINAL, &example.variables, &factors, 120);
    inference.infer();

    let log_z = inference.log_z();
    println!("Overall score = {}", (factors.iter().map(|f| f.score_assignment(&target_assignment)).sum::<f64>() - log_z).exp());
    
    for var in &example.variables {
        println!("\n***********");
        println!("VariableInfo (val={}): id={} source={} link={} target={}",
            var.get_label_value().idx == 1,
            var.get_id(),
            example.get_source(var).label,
            example.get_edge(var).label,
            example.get_target(var).label
        );
        let mut true_assignment = target_assignment.clone();
        let mut false_assignment = target_assignment.clone();
        true_assignment.insert(var.get_id(), model.annotator.var_domain.encode_value(&true));
        false_assignment.insert(var.get_id(), model.annotator.var_domain.encode_value(&false));

        let true_assignment = Rc::new(true_assignment);
        let false_assignment = Rc::new(false_assignment);

        println!("P[sm(var=true)]  = {:.9}", (factors.iter().map(|f| f.score_assignment(&true_assignment)).sum::<f64>() - log_z).exp());
        println!("P[sm(var=false)] = {:.9}", (factors.iter().map(|f| f.score_assignment(&false_assignment)).sum::<f64>() - log_z).exp());

        let mut true_score = 0.0;
        let mut false_score = 0.0;
                
        for factor in &factors {
            if factor.touch(var) {
                let true_fscore = factor.score_assignment(&true_assignment);
                let false_fscore = factor.score_assignment(&false_assignment);

                true_score += true_fscore;
                false_score += false_fscore;

                let con = MRRDebugContainer {
                    var: var.clone(),
                    target_assignment: Rc::clone(&rc_target_assignment),
                    true_assignment: Rc::clone(&true_assignment),
                    false_assignment: Rc::clone(&false_assignment),
                    tf_domain: Rc::clone(&rc_tfdomain),
                    pk_pairwise_domain: Rc::clone(&rc_pk_pairwise_domain),
                    dup_pairwise_domain: Rc::clone(&rc_dup_pairwise_domain),
                    cooccur_domain: Rc::clone(&rc_cooccur_domain),
                    example: Rc::clone(&rc_example)
                };

                factor.debug(&con);
            }
        }

        println!("\t=> un-norm score = {}", true_score);
    }
}