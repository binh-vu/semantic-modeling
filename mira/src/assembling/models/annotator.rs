use rdb2rdf::models::semantic_model::SemanticModel;
use gmtk::prelude::*;
use algorithm::data_structure::graph::Graph;
use std::collections::HashMap;
use assembling::models::example::*;
use assembling::models::variable::*;
use algorithm::data_structure::graph::Node;

use assembling::features::*;
use assembling::auto_label;
use assembling::features::node_prob::NodeProb;
use std::path::Path;
use std::collections::HashSet;
use std::sync::Arc;
use settings::Settings;

/// Annotating input for Markov Random Field model
pub struct Annotator<'a> {
    pub dataset: &'a str,
    pub workdir: &'a Path,
    pub sm_index: HashMap<String, usize>,
    pub sms: &'a [SemanticModel],
    pub train_sms: Vec<&'a SemanticModel>,
    pub var_domain: BooleanVectorDomain,
    pub max_permutation: usize,

    // store object features for annotating
    pub stats_predicate: Arc<StatsPredicate>,
    pub statistic: Statistic,
    pub node_prob: NodeProb,
    pub primary_key: PrimaryKey,
    pub cardinality: CardinalityFeatures,
    pub attr_scope: AttributeScope,
    pub local_structure: LocalStructure,
    pub cooccurrence: CooccurrenceFeatures,
    pub stype_assistant: &'a STypeAssistant,
}

impl<'a> Default for Annotator<'a> {
    fn default() -> Annotator<'a> {
        unimplemented!()
    }
}

impl<'a> Annotator<'a> {
    pub fn new(dataset: &'a str, workdir: &'a Path,
               sms: &'a [SemanticModel], train_sm_idxs: &[usize],
               stype_assistant: &'a STypeAssistant,
               primary_key: PrimaryKey, mut cardinality: CardinalityFeatures) -> Annotator<'a> {
        let sm_index: HashMap<String, usize> = sms.iter().enumerate().map(|(i, sm)| (sm.id.clone(), i)).collect();
        let train_sms = train_sm_idxs.iter().map(|&i| &sms[i]).collect::<Vec<_>>();
        let statistic = Statistic::new(&train_sms);
        let stats_predicate = Arc::new(StatsPredicate::new(&train_sms));
        let node_prob = NodeProb::new(Arc::clone(&stats_predicate));
        let attr_scope = AttributeScope::new(sms);
        cardinality.set_sm_index(sms);
        let cooccurrence = CooccurrenceFeatures::new(&train_sms);
        let local_structure = LocalStructure::new(&train_sms, &primary_key);

        unsafe { TRIPLE_VAR_DOMAIN = Some(BooleanVectorDomain::new()); }
        Annotator {
            dataset,
            sms,
            sm_index,
            workdir,
            train_sms,
            var_domain: BooleanVectorDomain::new(),
            max_permutation: Settings::get_instance().learning.max_permutation,
            statistic,
            stats_predicate,
            node_prob,
            primary_key,
            cardinality,
            attr_scope,
            cooccurrence,
            local_structure,
            stype_assistant
        }
    }

    pub fn clone(&self) -> Annotator<'a> {
        Annotator {
            dataset: self.dataset,
            sms: self.sms,
            sm_index: self.sm_index.clone(),
            workdir: self.workdir,
            train_sms: self.train_sms.clone(),
            var_domain: BooleanVectorDomain::new(),
            max_permutation: self.max_permutation,
            statistic: self.statistic.clone(),
            stats_predicate: Arc::clone(&self.stats_predicate),
            node_prob: self.node_prob.clone(),
            primary_key: self.primary_key.clone(),
            attr_scope: self.attr_scope.clone(),
            cardinality: self.cardinality.clone(),
            cooccurrence: self.cooccurrence.clone(),
            local_structure: self.local_structure.clone(),
            stype_assistant: self.stype_assistant
        }
    }

    pub fn reload(&mut self) {
        self.node_prob.reload(self.workdir);
    }

    pub fn train(&mut self, examples: &mut [MRRExample])  {
        for example in examples.iter_mut() {
            let sm = &self.sms[example.sm_idx];
            let graph = &example.graph;

            for n in graph.iter_nodes() {
                if n.is_data_node() {
                    let edge = n.first_incoming_edge(graph).unwrap();
                    let source = edge.get_source_node(graph);
                    let mut p_link_given_so: Option<f32> = None;

                    for stype in &sm.get_attr_by_label(&n.label).semantic_types {
                        if stype.class_uri == source.label && stype.predicate == edge.label {
                            p_link_given_so = Some(stype.score as f32);
                            break;
                        }
                    }

                    example.node_features[n.id].stype_score = p_link_given_so;
                    continue;
                }
            }
        }

        self.node_prob.train(self.dataset, self.workdir, examples);
    }

    pub fn create_labeled_mrr_example(&self, sm_id: &str, g: Graph) -> Option<MRRExample> {
        let sm_idx = self.sm_index[sm_id];
        let mrr_label;

        match auto_label::label(&self.sms[sm_idx].graph, &g, self.max_permutation) {
            None => {
                return None;
            },
            Some(r) => {
                mrr_label = r;
            }
        }

        let n_edges = g.n_edges;
        let mut mrr_example = MRRExample::new(sm_idx, g, self.attr_scope.does_sm_has_hierachy(sm_idx), Some(mrr_label));

        let ptr = &mut mrr_example as *mut MRRExample;
        let edge2label = unsafe { &(&mut *ptr).label.as_ref().unwrap().edge2label };

        for eid in 0..n_edges {
            mrr_example.add_var(eid, self.var_domain.get_value(edge2label[eid] as usize), eid);
        }

        Some(mrr_example)
    }

    pub fn create_mrr_example(&self, sm_id: &str, g: Graph) -> MRRExample {
        let sm_idx = self.sm_index[sm_id];
        let n_edges = g.n_edges;
        let mut mrr_example = MRRExample::new(sm_idx, g, self.attr_scope.does_sm_has_hierachy(sm_idx), None);

        let edge2label = vec![1; n_edges];
        for eid in 0..n_edges {
            mrr_example.add_var(eid, self.var_domain.get_value(edge2label[eid] as usize), eid);
        }

        mrr_example
    }

    pub fn annotate(&self, example: &mut MRRExample, tf_domain: &TFDomain) {
        self.pre_annotate(example);

        let builders = self.get_observed_features(example);
        for i in 0..builders.len() {
            example.observed_edge_features[i] = builders[i].create_tensor(tf_domain);
        }
    }

    /// This is a first phase, where each example can be annotate individually
    /// mainly to store/cached features
    pub fn pre_annotate(&self, example: &mut MRRExample) {
        let sm = &self.sms[example.sm_idx];

        // do this for loop first, so we have features for node_prob
        {
            let graph = &example.graph;
            for n in graph.iter_data_nodes() {
                let edge = n.first_incoming_edge(graph).unwrap();
                let source = edge.get_source_node(graph);
                let mut p_link_given_so: Option<f32> = None;

                for stype in &sm.get_attr_by_label(&n.label).semantic_types {
                    if stype.class_uri == source.label && stype.predicate == edge.label {
                        p_link_given_so = Some(stype.score as f32);
                        break;
                    }
                }

                example.node_features[n.id].stype_score = p_link_given_so;
            }
        }

        self.node_prob.update_node_features(example);
        let graph = &example.graph;

        for n in graph.iter_nodes() {
            let numbered_edges = StatsPredicate::numbering_edge_labels(n.iter_outgoing_edges(graph));

            for e in n.iter_outgoing_edges(graph) {
                let target = graph.get_node_by_id(e.target_id);
                let mut p_triple = None;
                let mut p_link_given_so = None;

                if target.is_class_node() {
                    p_link_given_so = Some(self.statistic.p_l_given_so(&n.label, &e.label, &target.label, 0.5));
                    p_triple = Some(p_link_given_so.unwrap() * example.node_features[n.id].node_prob.unwrap() *
                        example.node_features[target.id].node_prob.unwrap());
                } else {
                    let target_stypes = &sm.get_attr_by_label(&target.label).semantic_types;
                    let n_target_stypes = target_stypes.len();

                    for (i, stype) in target_stypes.iter().enumerate() {
                        if stype.class_uri == n.label && stype.predicate == e.label {
                            // data node, p_link = score of semantic type
                            p_link_given_so = Some(stype.score);

                            example.link_features[e.id].delta_stype_score = Some(if i == 0 && n_target_stypes > 1 {
                                stype.score - target_stypes[1].score
                            } else {
                                stype.score - target_stypes[0].score
                            });
                            example.link_features[e.id].ratio_stype_score = Some(stype.score / target_stypes[0].score);
                            example.link_features[e.id].stype_order = Some(i);
                            break;
                        }
                    }

                    if p_link_given_so.is_some() {
                        p_triple = Some(p_link_given_so.unwrap() * example.node_features[n.id].node_prob.unwrap());
                    }
                }

                example.link_features[e.id].p_link_given_s = Some(self.statistic.p_l_given_s(&n.label, &e.label, 0.01));
                example.link_features[e.id].p_triple = p_triple;
                example.link_features[e.id].p_link_given_so = p_link_given_so;
                example.link_features[e.id].multi_val_prob = self.stats_predicate.prob_multi_val(&e.label, numbered_edges[&e.id]);
            }
        }
    }

    /// Use in training to build observed features domain from training data; if the input domain is optional
    /// the annotator will construct a new domain from given examples; otherwise, they are untouched
    pub fn train_annotate(&self, examples: &mut [MRRExample]) -> TFDomain {
        let mut builders = Vec::with_capacity(examples.iter().map(|e| e.variables.len()).sum());
        for example in examples.iter_mut() {
            self.pre_annotate(example);
            builders.append(&mut self.get_observed_features(example));
        }

        let new_tf_domain = ObservedFeaturesBuilder::<String>::create_domain(&mut builders);
        let mut counter = 0;
        for example in examples.iter_mut() {
            for _i in 0..example.variables.len() {
                example.observed_edge_features[_i] = builders[counter].create_tensor(&new_tf_domain);
                counter += 1;
            }
        }

        return new_tf_domain;
    }

    pub fn get_sm(&self, sm_id: &str) -> &SemanticModel {
        &self.sms[self.sm_index[sm_id]]
    }

    fn get_observed_features(&self, example: &mut MRRExample) -> Vec<ObservedFeaturesBuilder<String>> {
        let graph = &example.graph;
        let mut builders = Vec::with_capacity(example.variables.len());

        for (var, vfeature) in example.variables.iter().zip(example.link_features.iter()) {
            let target: &Node = example.get_target(var);
            let source = example.get_source(var);
            let edge = example.get_edge(var);

            let mut builder = ObservedFeaturesBuilder::<String>::new();
            if let Some(p_link_given_so) = vfeature.p_link_given_so {
                if target.is_data_node() {
                    let name = format!("{}---{}", source.label, edge.label);
                    builder += (format!("{}=true.p_semantic_type", name), p_link_given_so.max(0.01));
                    builder += (format!("{}=false.p_semantic_type", name), (1.0 - p_link_given_so).max(0.01));
                    builder += (format!("{}=true.delta_p_semantic_type", name), vfeature.delta_stype_score.unwrap());
                    builder += (format!("{}=false.delta_p_semantic_type", name), (1.0 - vfeature.delta_stype_score.unwrap()).max(0.01));

                    // builder += (format!("{}=true.ratio_p_semantic_type", name), 1.0 / vfeature.ratio_stype_score.unwrap());
                    // builder += (format!("{}=false.ratio_p_semantic_type", name), vfeature.ratio_stype_score.unwrap());
                    builder += (format!("{}-order={}", name, vfeature.stype_order.unwrap()), 1.0);
                }

                builder += (format!("{}---{}=true.p_l_given_s", source.label, edge.label), vfeature.p_link_given_s.unwrap().max(0.01));
                builder += (format!("{}---{}=false.p_l_given_s", source.label, edge.label), (1.0 - vfeature.p_link_given_s.unwrap()).max(0.01));
                builder += (format!("{}---{}=true.p_triple", source.label, edge.label), vfeature.p_triple.unwrap().max(0.01));
                builder += (format!("{}---{}=false.p_triple", source.label, edge.label), (1.0 - vfeature.p_triple.unwrap()).max(0.01));
            }

            // if let Some(multi_val_prob) = vfeature.multi_val_prob {
            //     builder += ("multi_val_prob".to_owned(), multi_val_prob.max(0.01));
            //     builder += ("single_val_prob".to_owned(), (1.0 - multi_val_prob).max(0.01));
            // }

            if target.is_class_node() && target.iter_siblings_with_index(graph, &example.sibling_index).all(|n| n.is_class_node()) {
                builder += (format!("class={}.source_node_no_data_child", source.label), 1.0);
            }

            if target.is_class_node() && example.sibling_index.get_n_siblings(target) == 0 {
                builder += (format!("class={}.no_siblings", source.label), 1.0);

                if source.first_parent(graph).is_none() {
                    builder += (format!("class={}.no_parent_&_siblings", source.label), 1.0);
                }
            }

            builder += (String::from("prior"), 1.0);
            builders.push(builder);
        }

        builders
    }
}