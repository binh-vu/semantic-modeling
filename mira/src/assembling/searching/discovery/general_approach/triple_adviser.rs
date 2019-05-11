use utils::set_has;
use std::collections::{ HashMap, HashSet };
use rdb2rdf::models::semantic_model::*;
use rdb2rdf::ontology::ont_graph::OntGraph;

pub trait TripleAdviser {
    /// Get all possible (predicate, object)s that a subject can have
    fn get_pred_objs(&mut self, subject: &String) -> Vec<(String, String)>;
    /// Get all possible (subject, predicate)s that a object can belong to
    fn get_subj_preds(&mut self, object: &String) -> Vec<(String, String)>;
}

pub struct OntologyTripleAdviser<'a> {
    ont_graph: &'a OntGraph,
    class_nodes: Vec<String>,
    data_nodes: HashMap<String, Vec<(String, String)>>
}

impl<'a> OntologyTripleAdviser<'a> {
    pub fn new(ont_graph: &'a OntGraph, attributes: &[Attribute]) -> OntologyTripleAdviser<'a> {
        OntologyTripleAdviser {
            class_nodes: ont_graph.get_potential_class_node_uris().to_owned(),
            ont_graph,
            data_nodes: attributes.iter().map(|attr| (
                    attr.label.clone(),
                    attr.semantic_types.iter()
                        .map(|st| (st.class_uri.clone(), st.predicate.clone()))
                        .collect::<Vec<_>>()
                )).collect::<HashMap<_, _, _>>()
        }
    }
}

impl<'a> TripleAdviser for OntologyTripleAdviser<'a> {
    fn get_pred_objs(&mut self, subject: &String) -> Vec<(String, String)> {
        if self.data_nodes.contains_key(subject) {
            Vec::new()
        } else {
            self.class_nodes.iter()
                .flat_map(|object| {
                    self.ont_graph.get_possible_predicates(subject, object)
                        .map(|predicate| (predicate.uri.clone(), object.clone()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        }
    }

    fn get_subj_preds(&mut self, object: &String) -> Vec<(String, String)> {
        if self.data_nodes.contains_key(object) {
            // this is data node, use pre-defined semantic types
            self.data_nodes[object].clone()
        } else {
            self.class_nodes.iter()
                .flat_map(|subject| {
                    self.ont_graph.get_possible_predicates(subject, object)
                        .map(|predicate| (subject.clone(), predicate.uri.clone()))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        }
    }
}

pub struct EmpiricalTripleAdviser {
    subj_preds: HashMap<String, Vec<(String, String)>>,
    pred_objs: HashMap<String, Vec<(String, String)>>
}

impl EmpiricalTripleAdviser {
    pub fn new(train_sms: &[&SemanticModel], attributes: &[Attribute], max_candidates: usize) -> EmpiricalTripleAdviser {
        let mut adviser = EmpiricalTripleAdviser {
            subj_preds: Default::default(),
            pred_objs: Default::default(),
        };

        for attr in attributes {
            adviser.pred_objs.insert(attr.label.clone(), Vec::new());
            adviser.subj_preds.insert(
                attr.label.clone(), 
                attr.semantic_types.iter()
                    .map(|st| (st.class_uri.clone(), st.predicate.clone()))
                    .collect::<Vec<_>>()
            );
        }

        let mut subj_preds: HashMap<String, HashSet<(&str, &str)>> = Default::default();
        let mut pred_objs: HashMap<String, HashSet<(&str, &str)>> = Default::default();

        for sm in train_sms {
            for n in sm.graph.iter_class_nodes() {
                for e in n.iter_incoming_edges(&sm.graph) {
                    let source = e.get_source_node(&sm.graph);

                    if !subj_preds.contains_key(&n.label) {
                        subj_preds.insert(n.label.clone(), Default::default());
                        adviser.subj_preds.insert(n.label.clone(), Default::default());
                    }

                    if !subj_preds[&n.label].contains(&(&source.label, &e.label)) {
                        subj_preds.get_mut(&n.label).unwrap().insert((&source.label, &e.label));
                        adviser.subj_preds.get_mut(&n.label).unwrap().push((source.label.clone(), e.label.clone()));
                    }
                }
                for e in n.iter_outgoing_edges(&sm.graph) {
                    let target = e.get_target_node(&sm.graph);
                    if target.is_class_node() {
                        if !pred_objs.contains_key(&n.label) {
                            pred_objs.insert(n.label.clone(), Default::default());
                            adviser.pred_objs.insert(n.label.clone(), Default::default());
                        }

                        if !pred_objs[&n.label].contains(&(&e.label, &target.label)) {
                            pred_objs.get_mut(&n.label).unwrap().insert((&e.label, &target.label));
                            adviser.pred_objs.get_mut(&n.label).unwrap().push((e.label.clone(), target.label.clone()));
                        }
                    }
                }
            }
        }

        for val in adviser.subj_preds.values_mut() {
            val.truncate(max_candidates);
        }

        for val in adviser.pred_objs.values_mut() {
            val.truncate(max_candidates);
        }

        adviser
    }
}

impl TripleAdviser for EmpiricalTripleAdviser {
    fn get_pred_objs(&mut self, subject: &String) -> Vec<(String, String)> {
        if !self.pred_objs.contains_key(subject) {
            Vec::new()
        } else {
            self.pred_objs[subject].clone()
        }
    }

    fn get_subj_preds(&mut self, object: &String) -> Vec<(String, String)> {
        if !self.subj_preds.contains_key(object) {
            Vec::new()
        } else {
            self.subj_preds[object].clone()
        }
    }
}