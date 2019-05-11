use assembling::searching::banks::data_structure::int_graph::IntGraph;
use rdb2rdf::models::semantic_model::SemanticModel;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::generate_candidate_attr_mapping;
use assembling::searching::banks::attributes_mapping::mapping_score::mohsen_mapping_score;
use assembling::searching::banks::attributes_mapping::eval_mapping_score::evaluate_attr_mapping;
use fnv::FnvHashMap;
use im::HashSet as IHashSet;
use im::OrdSet as IOrdSet;
use fnv::FnvHashSet;
use im::HashMap as IHashMap;
use std::collections::HashMap;
use assembling::features::Statistic;
use assembling::features::StatsPredicate;
use rdb2rdf::models::semantic_model::SemanticType;
use rusty_machine::prelude::*;
use rusty_machine::learning::lin_reg::LinRegressor;
use rusty_machine::analysis::score::neg_mean_squared_error;
use assembling::searching::banks::data_structure::int_graph::IntEdge;
use rusty_machine::learning::logistic_reg::LogisticRegressor;
use rusty_machine::learning::optim::grad_desc::GradientDesc;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::MappingCandidate;
use assembling::searching::banks::attributes_mapping::mrr::learn_mrr::make_oracle_mapping;


pub struct TrainingData {
    x_train: Matrix<f64>,
    y_train: Vector<f64>,
    x_test: Matrix<f64>,
    y_test: Vector<f64>,
}


pub fn get_train_data(int_graph: &IntGraph, train_sms: &[&SemanticModel], test_sms: &[&SemanticModel], binarize_y: bool) -> (FeatureExtractor, TrainingData) {
    let branching_factor = 50;
    let feature_extractor = FeatureExtractor::new(train_sms);

    let mut x_train = Vec::new();
    let mut y_train = Vec::new();

    let mut x_test = Vec::new();
    let mut y_test = Vec::new();

    for (i, sm) in train_sms.iter().chain(test_sms.iter()).enumerate() {
        // don't need to adapt int_graph to sm, because its attributes are already included in training
        let mut attr_mappings = generate_candidate_attr_mapping(int_graph, &sm.attrs, branching_factor, &mut mohsen_mapping_score);

        if i < train_sms.len() {
            // only train have gold mapping
            attr_mappings.push(make_oracle_mapping(int_graph, sm));
        }

        for attr_mapping in &attr_mappings {
            let eval_attr_mapping = evaluate_attr_mapping(int_graph, sm, attr_mapping);
            let repopulated_attr_mapping = repopulate_attr_mapping(int_graph, sm, attr_mapping);

            if i < train_sms.len() {
                x_train.extend(&feature_extractor.extract_features(int_graph, &repopulated_attr_mapping));
                if binarize_y {
                    y_train.push((eval_attr_mapping.precision == 1.0) as usize as f64);
                } else {
                    y_train.push(eval_attr_mapping.precision as f64);
                }
            } else {
                x_test.extend(&feature_extractor.extract_features(int_graph, &repopulated_attr_mapping));
                if binarize_y {
                    y_test.push((eval_attr_mapping.precision == 1.0) as usize as f64);
                } else {
                    y_test.push(eval_attr_mapping.precision as f64);
                }
            }
        }
    }

    (
        feature_extractor,
        TrainingData {
            x_train: Matrix::new(y_train.len(), 4, x_train),
            y_train: Vector::new(y_train),
            x_test: Matrix::new(y_test.len(), 4, x_test),
            y_test: Vector::new(y_test)
        }
    )
}


pub fn repopulate_attr_mapping<'a>(int_graph: &'a IntGraph, sm: &'a SemanticModel, attr_mapping: &FnvHashMap<usize, usize>) -> IHashMap<usize, (&'a SemanticType, &'a IntEdge)> {
    sm.attrs.iter()
        .filter(|attr| attr_mapping.contains_key(&attr.id))
        .map(|attr| {
            let e = int_graph.graph.get_edge_by_id(attr_mapping[&attr.id]);
            let n = e.get_source_node(&int_graph.graph);

            for st in &attr.semantic_types {
                if st.class_uri == n.label && st.predicate == e.label {
                    return (attr.id, Some((st, e)));
                }
            }

            return (attr.id, None)
        })
        .filter(|(k, v)| v.is_some())
        .map(|(k, v)| (k, v.unwrap()))
        .collect::<IHashMap<_, _>>()
}


pub struct LinearRegressionModel {
    model: LinRegressor,
    feature_extractor: FeatureExtractor
}


pub struct LogisticRegressionModel {
    model: LogisticRegressor<GradientDesc>,
    feature_extractor: FeatureExtractor
}

impl LinearRegressionModel {
    pub fn new(int_graph: &IntGraph, train_sms: &[&SemanticModel], test_sms: &[&SemanticModel]) -> LinearRegressionModel {
        let (feature_extractor, train_data) = get_train_data(int_graph, train_sms, test_sms, false);

        let mut lin_mod = LinRegressor::default();
        lin_mod.train(&train_data.x_train, &train_data.y_train).unwrap();

        let y_train_pred = lin_mod.predict(&train_data.x_train).unwrap();
        let y_test_pred = lin_mod.predict(&train_data.x_test).unwrap();

        let train_mse = neg_mean_squared_error(
            &Matrix::new(y_train_pred.size(), 1, y_train_pred),
            &Matrix::new(train_data.y_train.size(), 1, train_data.y_train));
        let test_mse = neg_mean_squared_error(
            &Matrix::new(y_test_pred.size(), 1, y_test_pred),
            &Matrix::new(train_data.y_test.size(), 1, train_data.y_test));

        info!("mse[train] = {:.5}", train_mse);
        info!("mse[test] = {:.5}", test_mse);

        LinearRegressionModel {
            model: lin_mod,
            feature_extractor
        }
    }

    pub fn mapping_score(&self, int_graph: &IntGraph, attr_mappings: &[MappingCandidate]) -> Vec<f64> {
        let mut features = Vec::new();
        for mc in attr_mappings.iter() {
            features.extend(&self.feature_extractor.extract_features(int_graph, &mc.mapping));
        }

        let input = Matrix::new(attr_mappings.len(), 4, features);
        let y_pred = self.model.predict(&input);
        y_pred.unwrap().into_vec()
    }
}

impl LogisticRegressionModel {
    pub fn new(int_graph: &IntGraph, train_sms: &[&SemanticModel], test_sms: &[&SemanticModel]) -> LogisticRegressionModel {
        let (feature_extractor, train_data) = get_train_data(int_graph, train_sms, test_sms, true);

        let mut model = LogisticRegressor::default();
        model.train(&train_data.x_train, &train_data.y_train).unwrap();

        let y_train_pred = model.predict(&train_data.x_train).unwrap();
        let y_test_pred = model.predict(&train_data.x_test).unwrap();


        LogisticRegressionModel {
            model,
            feature_extractor
        }
    }

    pub fn mapping_score(&self, int_graph: &IntGraph, attr_mappings: &[MappingCandidate]) -> Vec<f64> {
        let mut features = Vec::new();
        for mc in attr_mappings.iter() {
            features.extend(&self.feature_extractor.extract_features(int_graph, &mc.mapping));
        }

        let input = Matrix::new(attr_mappings.len(), 4, features);
        let y_pred = self.model.predict(&input);
        y_pred.unwrap().into_vec()
    }
}


type Feature = [f64; 4];


pub struct FeatureExtractor {
    stats_predicate: StatsPredicate
}

impl FeatureExtractor {
    pub fn new(train_sms: &[&SemanticModel]) -> FeatureExtractor {
        FeatureExtractor {
            stats_predicate: StatsPredicate::new(train_sms)
        }
    }

    fn extract_features(&self, int_graph: &IntGraph, attr_mapping: &IHashMap<usize, (&SemanticType, &IntEdge)>) -> Feature {
        // helper
        let mut source2attrs: FnvHashMap<usize, Vec<usize>> = Default::default();
        for (attr_id, (st, ie)) in attr_mapping {
            source2attrs.entry(ie.source_id).or_insert(Vec::new()).push(*attr_id);
        }
        let nodes: FnvHashSet<usize> = attr_mapping.values()
            .flat_map(|m| vec![m.1.source_id, m.1.target_id])
            .collect::<FnvHashSet<_>>();
        let n_nodes = nodes.len();

        // global coherence
        let mut tags: HashMap<&str, FnvHashSet<usize>> = Default::default();
        for (stype, edge) in attr_mapping.values() {
            for tag in &edge.get_source_node(&int_graph.graph).data.tags {
                tags.entry(&tag).or_insert(Default::default()).insert(edge.source_id);
            }
            for tag in &edge.get_target_node(&int_graph.graph).data.tags {
                tags.entry(&tag).or_insert(Default::default()).insert(edge.target_id);
            }
        }
        let global_coherence = tags.values().map(|x| x.len()).max().unwrap() as f64 / n_nodes as f64;

        // local coherence
        let mut local_coherence = 0.0;
        for attrs in source2attrs.values() {
            // tags => set of edge ids
            let mut tags: HashMap<&str, FnvHashSet<usize>> = Default::default();
            for a in attrs {
                for tag in attr_mapping[a].1.data.tags.iter() {
                    tags.entry(tag).or_insert(Default::default()).insert(attr_mapping[a].1.id);
                }
            }

            local_coherence += (tags.values().max_by_key(|v| v.len()).unwrap().len() as f64).ln() / attrs.len() as f64;
        }

        // unique column estimation
        let mut unique_column_score = 0.0;
        for attrs in source2attrs.values() {
            let numbered_edges = StatsPredicate::numbering_edge_labels(attrs.iter().map(|a| attr_mapping[a].1));
            for (eid, c) in numbered_edges {
                unique_column_score += match self.stats_predicate.prob_multi_val(&int_graph.graph.get_edge_by_id(eid).label, c) {
                    None => {
                        if c == 1 {
                            // may be because it's = 1
                             0.9_f64.ln()
                        } else {
                            // default when it's > 1, normally it's not duplicated
                            0.1_f64.ln()
                        }
                    },
                    Some(prob_x_multival) => {
                        (1.0 - prob_x_multival).ln() as f64
                    }
                };
            }
        }

        // confidence
        let mut confidence = attr_mapping.values().map(|(st, ie)| st.score as f64).sum::<f64>() / attr_mapping.len() as f64;

        // size_reduction
        let size_reduction = (2 * attr_mapping.len() - n_nodes) as f64 / attr_mapping.len() as f64;

        [confidence, local_coherence, size_reduction, global_coherence]
    }
}


pub fn compute_gold_mapping(int_graph: &IntGraph, sm: &SemanticModel) -> FnvHashMap<usize, usize> {
    let mut source2attrs: FnvHashMap<usize, Vec<usize>> = Default::default();
    let mut source2isources: FnvHashMap<usize, IOrdSet<usize>> = Default::default();

    for attr in &sm.attrs {
        let e = sm.graph.get_node_by_id(attr.id).first_incoming_edge(&sm.graph).unwrap();
        let source = e.get_source_node(&sm.graph);

        source2attrs.entry(e.source_id).or_insert(Vec::new()).push(attr.id);

        // find all possible edges that can hook up with e (in attr)
        let mut corresponding_source_ids = IOrdSet::new();
        for int_node in int_graph.graph.iter_nodes_by_label(&source.label) {
            for ie in int_node.iter_outgoing_edges(&int_graph.graph) {
                if ie.label == e.label && ie.data.tags.contains(&sm.id) {
                    corresponding_source_ids.insert(ie.source_id);
                }
            }
        }

        if source2isources.contains_key(&e.source_id) {
            corresponding_source_ids = corresponding_source_ids.intersection(source2isources.remove(&e.source_id).unwrap());
        }

        source2isources.insert(e.source_id, corresponding_source_ids);
    }

    // find the correct mapping from source id to int source id
    let mut source2isources: Vec<(usize, IOrdSet<usize>)> = source2isources.into_iter().collect::<Vec<_>>();
    source2isources.sort_by_key(|v| v.1.len());
    let choices = compute_gold_mapping_select_(&source2isources.iter().map(|v| v.1.clone()).collect::<Vec<_>>())
        .expect("Must not be None");

    let mut source2isources: FnvHashMap<usize, usize> = source2isources.iter().zip(choices.iter())
        .map(|(v, &c)| (v.0, c))
        .collect();

    let mut attr_mapping: FnvHashMap<usize, usize> = Default::default();

    for attr in &sm.attrs {
        let e = sm.graph.get_node_by_id(attr.id).first_incoming_edge(&sm.graph).unwrap();
        println!("[DEBUG] source2isources[&e.source_id] = {}", source2isources[&e.source_id]);
        let ie = int_graph.graph.get_node_by_id(source2isources[&e.source_id])
            .iter_outgoing_edges(&int_graph.graph)
            .find(|ie| {
                if ie.label == e.label && ie.data.tags.contains(&sm.id) {
                    return ie.get_target_node(&int_graph.graph).is_data_node();
                }

                return false;
            }).unwrap();

        attr_mapping.insert(attr.id, ie.id);
    }

    return attr_mapping;
}

fn compute_gold_mapping_select_(options: &[IOrdSet<usize>]) -> Option<Vec<usize>> {
    if options.len() == 0 {
        return Some(Vec::new());
    }

    if options.iter().any(|x| x.len() == 0) {
        return None;
    }

    for &choice in &options[0] {
        let new_options = options[1..].iter()
            .map(|x| x.without(&choice))
            .collect::<Vec<_>>();

        if let Some(mut rest_choices) = compute_gold_mapping_select_(&new_options) {
            rest_choices.insert(0, choice);
            return Some(rest_choices);
        }
    }

    return None;
}