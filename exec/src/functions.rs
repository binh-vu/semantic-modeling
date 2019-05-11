use serde_json;
use input::RustInput;
use mira::assembling::models::annotator::Annotator;
use mira::assembling::learning::gen_train_data::gen_train_data;
use std::fs::File;
use mira::settings::Settings;
use std::path::Path;
use std::io::BufWriter;
use mira::assembling::models::example::MRRExample;
use std::io::BufReader;
use mira::assembling::models::mrr::MRRModel;
use mira::assembling::predicting::predicting;
use experiment::evaluate;

pub fn create_train_data(input: &RustInput, file_prefix: &str) {
    let gen_data_method = &Settings::get_instance().learning.gen_data_method;
    let mut annotator: Annotator = input.get_annotator();
    let mut train_examples = gen_train_data(&input.get_train_sms()[..], annotator.clone(), &input.ont_graph, gen_data_method);
    let output_file = Path::new(&input.workdir).join(format!("{}.{}.json", file_prefix, gen_data_method.get_name()));

    serde_json::to_writer(BufWriter::new(File::create(&output_file).unwrap()), &train_examples).unwrap();
}

pub fn train_mrr<'a>(input: &'a RustInput, train_file: &str, test_file: &str) -> MRRModel<'a> {
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

    let model_file = workdir.join("model.bin");
    let mrr_model = MRRModel::train(annotator, &mut train_examples, &mut test_examples, &Settings::get_instance().mrf);
    mrr_model.serialize(&model_file);

    return mrr_model;
}

pub fn predict(input: &RustInput, mrr_model: &MRRModel) {
    let workdir = Path::new(&input.workdir);
    let evaluate_sms = input.get_test_sms();
    let pred_sms = predicting(&mrr_model, &evaluate_sms);
    evaluate(&evaluate_sms, &pred_sms, &workdir.join("result.csv"));
    serde_json::to_writer(File::create(&workdir.join("prediction.json")).unwrap(), &pred_sms).unwrap();
}