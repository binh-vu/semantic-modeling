use rdb2rdf::prelude::*;
use std::path::Path;
use std::collections::HashSet;
use std::collections::HashMap;
use settings::Settings;
use settings::conf_learning::*;
use settings::conf_search::DiscoverMethod;
use algorithm::data_structure::graph::Graph;
use assembling::models::annotator::Annotator;
use assembling::models::example::MRRExample;
use std::fs::File;
use serde_json;
use rayon::prelude::*;
use assembling::other_models::BayesModel;
use assembling::models::mrr::MRRModel;
use rand::SeedableRng;
use super::trial_and_error;
use super::iterative_mrr_approach;
use super::elimination;
use rand::StdRng;
use algorithm::random::random_choice::RandomChoice;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use algorithm::data_structure::unique_array::UniqueArray;
use assembling::searching::banks::MohsenWeightingSystem;


pub fn gen_train_data(discover_sms: &[&SemanticModel], annotator: Annotator, ont_graph: &OntGraph, gen_data_method: &GenTrainDataMethod) -> Vec<MRRExample> {
    let mut ignore_sids: HashSet<String> = Default::default();
    let mut data: HashMap<String, HashSet<String>> = Default::default();
    let mut examples: Vec<MRRExample> = Vec::new();

    for sm in discover_sms.iter() {
        data.insert(sm.id.clone(), Default::default());
    }

    match *gen_data_method {
        GenTrainDataMethod::TrialAndError { beam_width, max_candidates_per_round } => {
            let candidate_smss = discover_sms.par_iter()
                .map(|sm| {
                    let prob_sms = |graphs: Vec<Graph>| {
                        graphs.into_iter().map(|g| (g, 1.0)).collect::<Vec<_>>()
                    };

                    trial_and_error::generate_candidate_sms(prob_sms, &annotator, sm, ont_graph, max_candidates_per_round, beam_width)
                })
                .collect::<Vec<_>>();

            for (i, candidate_sms) in candidate_smss.into_iter().enumerate() {
                let sm = discover_sms[i];
                if ignore_sids.contains(&sm.id) {
                    continue;
                }

                if candidate_sms.len() == 0 {
                    // no new candidate sms
                    info!("No new candidate for source: {}", sm.id);
                    ignore_sids.insert(sm.id.clone());
                } else {
                    for (key, graph) in candidate_sms {
                        if !data[&sm.id].contains(&key) {
                            data.get_mut(&sm.id).unwrap().insert(key);
                            if let Some(example) = annotator.create_labeled_mrr_example(&sm.id, graph) {
                                examples.push(example);
                            }
                        }
                    }
                }
            }

            // === [DEBUG] DEBUG CODE START HERE ===
            // println!("[DEBUG] at gen_train_data.rs");
            // use debug_utils::*;
            // println!("[DEBUG] clear dir"); clear_dir();
            // draw_graphs(&examples.iter().map(|e| e.graph.clone()).collect::<Vec<_>>());
            // === [DEBUG] DEBUG CODE  END  HERE ===
        },
        GenTrainDataMethod::Elimination(ref elimination) => {
            let int_graph = match elimination.discover_method {
                DiscoverMethod::GeneralDiscovery(ref conf) => None,
                DiscoverMethod::ConstraintSpace(ref conf) =>Some(IntGraph::new(&annotator.train_sms))
            };

            let candidate_smss = discover_sms.par_iter()
                .map(|sm| {
                    elimination::generate_candidate_sms(&annotator, sm, elimination, int_graph.as_ref(), None)
                })
                .collect::<Vec<_>>();

            let max_n_examples = Settings::get_instance().learning.max_n_examples;
            for (i, candidate_sms) in candidate_smss.into_iter().enumerate() {
                let sm = discover_sms[i];
                if ignore_sids.contains(&sm.id) {
                    continue;
                }

                if candidate_sms.len() == 0 {
                    // no new candidate sms
                    info!("No candidate for source: {}", sm.id);
                    ignore_sids.insert(sm.id.clone());
                } else {
                    // if the number of examples is greater than let's say 300, random sampling it
                    let mut new_examples: Vec<_> = if candidate_sms.len() > max_n_examples {
                        info!("{}: Generated total {} examples, sampling it to: {} ones", sm.id, candidate_sms.len(), max_n_examples);
                        let rng = StdRng::from_seed([Settings::get_instance().manual_seed; 32]);
                        let mut random_choice = RandomChoice::new(rng);

                        let p = candidate_sms.get_ref_value().iter().map(|(g, s)| *s).collect::<Vec<_>>();
                        let mut choices = random_choice.random_choice_f64_without_replace(&p, max_n_examples);
                        choices.sort_unstable();

                        let mut graphs_w_scores = candidate_sms.get_value();
                        let mut tmp_graphs = Vec::with_capacity(max_n_examples);

                        for &i in choices.iter().rev() {
                            let g = graphs_w_scores.swap_remove(i).0;
                            tmp_graphs.push(g);
                        }

                        tmp_graphs.into_par_iter()
                            .map(|g| annotator.create_labeled_mrr_example(&sm.id, g))
                            .filter(|x| x.is_some())
                            .map(|x| x.unwrap())
                            .collect()
                    } else {
                        candidate_sms.get_value().into_par_iter()
                            .map(|(g, score)| annotator.create_labeled_mrr_example(&sm.id, g))
                            .filter(|x| x.is_some())
                            .map(|x| x.unwrap())
                            .collect()
                    };

                    examples.append(&mut new_examples);
                }
            }
        },
        GenTrainDataMethod::IterativeMRRApproach { n_iter, ref beam_settings, max_candidates_per_round } => {
            // create mrr model from default trainset
            let mut model = iterative_mrr_approach::get_default_model(annotator);

            for i in 0..(n_iter - 1) {
                info!("===========> Iter: {}", i);
                if i > 0 {
                    model = MRRModel::train(model.annotator, &mut examples, &mut Vec::new(), &Settings::get_instance().mrf);
                }

                let candidate_smss = discover_sms.par_iter()
                    .map(|sm| {
                        if ignore_sids.contains(&sm.id) {
                            return Default::default();
                        }

                        let prob_sms = |graphs: Vec<Graph>| model.predict_sm_probs(&sm.id, graphs);
                        return iterative_mrr_approach::generate_candidate_sms(prob_sms, &model.annotator, sm, ont_graph, max_candidates_per_round, beam_settings);
                    })
                    .collect::<Vec<_>>();

                for (i, candidate_sms) in candidate_smss.into_iter().enumerate() {
                    let sm = discover_sms[i];
                    if ignore_sids.contains(&sm.id) {
                        continue;
                    }

                    if candidate_sms.len() == 0 {
                        // no new candidate sms
                        info!("No new candidate for source: {}", sm.id);
                        ignore_sids.insert(sm.id.clone());
                    } else {
                        for (key, graph) in candidate_sms {
                            if !data[&sm.id].contains(&key) {
                                data.get_mut(&sm.id).unwrap().insert(key);
                                if let Some(example) = model.annotator.create_labeled_mrr_example(&sm.id, graph) {
                                    examples.push(example);
                                }
                            }
                        }
                    }
                }
            }
        },
        GenTrainDataMethod::GeneralBayesApproach { ref beam_settings, max_candidates_per_round } => {
            let bayes_model = BayesModel::new(&annotator);
            let candidate_smss = discover_sms.par_iter()
                .map(|sm| {
                    let prob_sms = |graphs: Vec<Graph>| bayes_model.predict_sm_probs(&sm.id, graphs);
                    return iterative_mrr_approach::generate_candidate_sms(prob_sms, &annotator, sm, ont_graph, max_candidates_per_round, beam_settings);
                })
                .collect::<Vec<_>>();

            for (i, candidate_sms) in candidate_smss.into_iter().enumerate() {
                let sm = discover_sms[i];
                if ignore_sids.contains(&sm.id) {
                    continue;
                }

                if candidate_sms.len() == 0 {
                    // no new candidate sms
                    info!("No new candidate for source: {}", sm.id);
                    ignore_sids.insert(sm.id.clone());
                } else {
                    for (key, graph) in candidate_sms {
                        if !data[&sm.id].contains(&key) {
                            data.get_mut(&sm.id).unwrap().insert(key);
                            if let Some(example) = annotator.create_labeled_mrr_example(&sm.id, graph) {
                                examples.push(example);
                            }
                        }
                    }
                }
            }
        }
        _ => panic!("Not implemented")
    }

    examples
}