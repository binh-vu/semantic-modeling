use super::input::*;
use std::path::*;
use mira::prelude::*;
use serde_json;
use serde_json::*;
use std::io::*;
use std::fs::*;
use prettytable::Table;
use gmtk::prelude::*;


fn float_option2json(val: &Option<f32>) -> Value {
    match val {
        None => Value::Null,
        Some(ref x) => Value::Number(Number::from_f64(*x as f64).unwrap())
    }
}

fn usize_option2json(val: &Option<usize>) -> Value {
    match val {
        None => Value::Null,
        Some(ref x) => Value::Number(Number::from(*x))
    }
}

pub fn write_features(input: &RustInput, model_file: &Path, train_file: &Path, test_file: &Path, output_file: &Path) {
    let mut model = MRRModel::deserialize(input.get_annotator(), model_file);
    let mut train_examples: Vec<MRRExample> = serde_json::from_reader(BufReader::new(File::open(train_file).unwrap())).unwrap();
    let mut test_examples: Vec<MRRExample> = serde_json::from_reader(BufReader::new(File::open(test_file).unwrap())).unwrap();
    
    for mut example in train_examples.iter_mut().chain(test_examples.iter_mut()) {
        example.deserialize();
    }
    // model.annotator.train(&mut train_examples);

    let n_train_examples = train_examples.len();
    let n_test_examples = test_examples.len();

    let mut train_rows = Vec::new();
    let mut test_rows = Vec::new();

    for (eno, mut example) in train_examples.iter_mut().enumerate().chain(test_examples.iter_mut().enumerate()) {
        model.annotator.annotate(&mut example, &model.tf_domain);

        for (i, var) in example.variables.iter().enumerate() {
            let row = json!({
                "provenance": format!("sid={}--eno={}--vid={}--{}---{}---{}", example.sm_idx, eno, i, example.get_source(var).label, example.get_edge(var).label, example.get_target(var).label),
                "p_triple": float_option2json(&example.link_features[i].p_triple),
                "p_link_given_so": float_option2json(&example.link_features[i].p_link_given_so),
                "multi_val_prob": float_option2json(&example.link_features[i].multi_val_prob),
                "stype_order": usize_option2json(&example.link_features[i].stype_order),
                "delta_stype_score": float_option2json(&example.link_features[i].delta_stype_score),
                "ratio_stype_score": float_option2json(&example.link_features[i].ratio_stype_score),
                "p_link_given_s": float_option2json(&example.link_features[i].p_link_given_s),
                "stype_score": float_option2json(&example.node_features[i].stype_score),
                "node_prob": float_option2json(&example.node_features[i].node_prob),
                "label": example.label.as_ref().unwrap().edge2label[var.id]
            });

            if eno < n_train_examples {
                train_rows.push(row);
            } else {
                test_rows.push(row);
            }
        }
    }
    
    let features = json!({
        "train_examples": train_rows,
        "test_examples": test_rows,
        "cooccurrence": model.annotator.cooccurrence,
        "stats_p_l_given_s": model.annotator.statistic.p_l_given_s,
        "stats_p_n": model.annotator.statistic.p_n,
    });

    let mut writer = BufWriter::new(File::create(output_file).unwrap());
    serde_json::to_writer_pretty(writer, &features).unwrap();
}

/// Print all weights in the model, so we can inspect their correctness
pub fn inspect_model(input: &RustInput, mrr_model: &MRRModel, output_file: &Path) {
    // because of not able to downcast trait object, here we really have to assume that we know the structure
    let weights = mrr_model.model.get_parameters();
    assert_eq!(weights.len(), 7, "|weights| != 7, some body have modified the model");

    let mut writer = File::create(output_file).unwrap();
    let triple_template_weight = weights[0].get_value().view(&[2, mrr_model.tf_domain.numel() as i64]);

    writer.write_all(b">>>> WRITE TRIPLE TEMPLATE WEIGHTS:\n").unwrap();
    let mut table = Table::new();
    table.add_row(row!["category", "false", "true"]);
    for i in 0..mrr_model.tf_domain.numel() {
        table.add_row(row![mrr_model.tf_domain.get_category(i), triple_template_weight.at((0, i as i64)).get_f64(), triple_template_weight.at((1, i as i64)).get_f64()]);
    }
    writer.write_all(table.to_string().as_bytes()).unwrap();

    writer.write_all(b">>>> WRITE ALL CHILDREN WEIGHTS:\n").unwrap();
    let all_children_weight = weights[1].get_value();
    let mut table = Table::new();
    table.add_row(row!["false", "true"]);
    table.add_row(row![all_children_weight.at(0).get_f64(), all_children_weight.at(1).get_f64()]);
    writer.write_all(table.to_string().as_bytes()).unwrap();

    writer.write_all(b">>>> WRITE PrimaryKEY WEIGHTS:\n").unwrap();
    let pk_weight = weights[2].get_value();
//    let mut table = Table::new();
//    table.add_row(row!["category", "false", "true"]);
//    for i in 0..mrr_model.tf_domain.numel() {
//        table.add_row(row![mrr_model.tf_domain.get_category(i), triple_template_weight.at((0, i as i64)), triple_template_weight.at((1, i as i64))]);
//    }
//
//    writer.write_all(table.to_string().as_bytes()).unwrap();
}