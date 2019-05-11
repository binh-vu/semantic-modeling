// `error_chain!` can recurse deeply
#![recursion_limit = "1024"]

extern crate mira;
extern crate algorithm;
extern crate rdb2rdf;

#[macro_use]
extern crate error_chain;
extern crate zmq;

#[macro_use]
extern crate serde_derive;
#[macro_use]
extern crate serde_json;
#[macro_use]
extern crate log;
extern crate serde_yaml;
extern crate csv;
extern crate env_logger;
extern crate gmtk;
extern crate fnv;
#[macro_use]
extern crate clap;
#[macro_use]
extern crate prettytable;
extern crate itertools;
extern crate im;

use std::path::Path;
use std::io::*;
use std::fs::*;
use std::str::FromStr;

use mira::prelude::*;
use helper::*;

mod input;
mod experiment;
mod debug_model;
mod debug_features;
mod helper;
mod debug;
mod debug_prediction;
mod api;
mod functions;

fn main() {
    env_logger::init();
    let mut timer = Timer::start();

    let app_m = clap_app!(sm =>
        (@arg input: -i +required +takes_value "input file")
        (@arg settings: -c +takes_value "settings file")
        (@subcommand train =>
            (about: "Train model using provided train/test set")
            (@arg train_file: -t +required +takes_value "train file")
            (@arg test_file: -p +required +takes_value "test file")
            (@arg run_prediction: -r +takes_value "run prediction [default=true]")
            (@arg run_map_prediction: -m +takes_value "run map prediction [default=true]")
        )
        (@subcommand gentrain =>
            (about: "Generate training examples")
            (@arg train_sms: -t +required +takes_value "semantic models we want to generate for. E.g: ['s00-s10', 's00-s10', 's01-s12'], inclusive range")
            (@arg file_prefix: -f +required +takes_value "prefix of file we want to write the examples to")
        )
        (@subcommand exec_func =>
            (about: "Run experiment function")
            (@arg model_file: -m +required +takes_value "model file")
            (@arg func: -f +required +takes_value "function name")
            (@arg arg_0: --arg_0 +takes_value "Argument 0")
            (@arg arg_1: --arg_1 +takes_value "Argument 1")
            (@arg arg_2: --arg_2 +takes_value "Argument 2")
            (@arg arg_3: --arg_3 +takes_value "Argument 3")
        )
        (@subcommand pred =>
            (about: "Run prediction")
            (@arg model_file: -m +required +takes_value "model file")
        )
        (@subcommand exp =>
            (about: "Run experiment")
            (@arg gentest: -g +takes_value "generate testset")
        )
        (@subcommand debug =>
            (about: "Run debug in general")
        )
        (@subcommand debug_prediction =>
            (about: "Run debug prediction")
            (@arg pred_file: -p +required +takes_value "prediction file")
        )
        (@subcommand debug_features =>
            (about: "Run debug features")
            (@arg model_file: -m +required +takes_value "model file")
            (@arg train_file: -t +required +takes_value "train file")
            (@arg test_file: -p +required +takes_value "test file")
            (@arg output_file: -o +required +takes_value "output file")
        )
        (@subcommand debug_model_with_provided_graph =>
            (about: "Run debug model with provided graph")
            (@arg model_file: -m +required +takes_value "model file")
            (@arg sm_id: -s +required +takes_value "id of semantic model")
            (@arg graph: -g +required +takes_value "the graph you want to evaluate")
        )
    ).get_matches();

    if app_m.is_present("settings") {
        let fsettings = app_m.value_of("settings").unwrap();
        let conf: input::Configuration = serde_yaml::from_reader(BufReader::new(File::open(fsettings).unwrap())).unwrap();
        Settings::update_instance(conf.settings);

        debug!("Current settings: {:?}", Settings::get_instance());
    }

    let finput = Path::new(app_m.value_of("input").unwrap());
    let mut input = input::RustInput::from_file(&finput);
    let tmp_workdir = input.workdir.clone();
    let workdir = Path::new(&tmp_workdir);

    match app_m.subcommand() {
        ("exec_func", Some(sub_m)) => {
            match sub_m.value_of("func").unwrap() {
                "experiment::run_predict_fixed" => {
                    let mrr_model = experiment::load_model(&input, sub_m.value_of("model_file").unwrap());
                    experiment::run_predict_fixed(
                        &input, &mrr_model,
                        &workdir.join(sub_m.value_of("arg_0").unwrap()),
                        &workdir.join(sub_m.value_of("arg_1").unwrap()),
                        &workdir.join(sub_m.value_of("arg_2").unwrap())
                    );
                },
                "experiment::run_simulated_interactive_modeling" => {
                    let mrr_model = experiment::load_model(&input, sub_m.value_of("model_file").unwrap());
                    experiment::run_simulated_interactive_modeling(&input, &mrr_model);
                },
                "experiment::run_exp_interactie_modeling" => {
                    experiment::run_exp_interactive_modeling(&input);
                },
                "debug_features::inspect_model" => {
                    let mrr_model = experiment::load_model(&input,sub_m.value_of("model_file").unwrap());
                    debug_features::inspect_model(&input, &mrr_model, &workdir.join(sub_m.value_of("arg_0").unwrap()));
                },
                "debug_model::debug_model" => {
                    let mrr_model = experiment::load_model(&input,sub_m.value_of("model_file").unwrap());
                    debug_model::debug_model(&input,
                         &mrr_model,
                         &workdir.join(sub_m.value_of("arg_0").unwrap()),
                         &workdir.join(sub_m.value_of("arg_1").unwrap()),
                         &workdir.join(sub_m.value_of("arg_2").unwrap()),
                         sub_m.value_of("arg_3").unwrap()
                    );
                },
                "debug::bank_method" => {
                    debug::bank_method(&mut input, sub_m.value_of("model_file").unwrap());
                },
                "func::create_train_data" => {
                    functions::create_train_data(
                        &input, sub_m.value_of("arg_0").unwrap());
                },
                "func::train_and_predict" => {
                    let mrr_model = functions::train_mrr(
                        &input, sub_m.value_of("arg_0").unwrap(), sub_m.value_of("arg_1").unwrap());
                    functions::predict(&input, &mrr_model);
                },
                "func::train" => {
                    functions::train_mrr(
                        &input, sub_m.value_of("arg_0").unwrap(), sub_m.value_of("arg_1").unwrap());
                },
                "func::predict" => {
                    let mrr_model = experiment::load_model(&input,sub_m.value_of("arg_0").unwrap());
                    functions::predict(&input, &mrr_model);
                },
                func_name => panic!("Invalid function: {}", func_name)
            }
        },
        ("gentrain", Some(sub_m)) => {
            experiment::run_gentrain(
                &input,
                &parse_sm_query(&input, sub_m.value_of("train_sms").unwrap()),
                sub_m.value_of("file_prefix").unwrap());
        },
        ("train", Some(sub_m)) => {
            experiment::run_train(&input,
                                  sub_m.value_of("train_file").unwrap(), sub_m.value_of("test_file").unwrap(),
                                  bool::from_str(sub_m.value_of("run_prediction").unwrap_or("true")).unwrap(),
                                  bool::from_str(sub_m.value_of("run_map_prediction").unwrap_or("true")).unwrap()
            );
        },
        ("pred", Some(sub_m)) => {
            let mrr_model = experiment::load_model(&input, sub_m.value_of("model_file").unwrap());
            experiment::run_predict(&input, &mrr_model);
        },
        ("exp", Some(sub_m)) => {
            let gentest = bool::from_str(sub_m.value_of("gentest").unwrap_or("false")).unwrap();
            experiment::run_exp(&input, gentest);
        },
        ("debug", Some(_)) => {
            debug::run(&input);
        },
        ("debug_prediction", Some(sub_m)) => {
            debug_prediction::run_rerank(&input,
                &workdir.join(sub_m.value_of("pred_file").unwrap()),
            )
        },
        ("debug_features", Some(sub_m)) => {
            debug_features::write_features(&input,
                &workdir.join(sub_m.value_of("model_file").unwrap()),
                &workdir.join(sub_m.value_of("train_file").unwrap()),
                &workdir.join(sub_m.value_of("test_file").unwrap()),
                &workdir.join(sub_m.value_of("output_file").unwrap()),
            )
        },
        ("debug_model_with_provided_graph", Some(sub_m)) => {
            debug_model::debug_model_provided_graph(&input,
                &workdir.join(sub_m.value_of("model_file").unwrap()),
                sub_m.value_of("sm_id").unwrap(),
                sub_m.value_of("graph").unwrap(),
            );
        },
        _ => panic!("No subcommand provided")
    }

    timer.lap_and_report("!! Finish the program !!");
}