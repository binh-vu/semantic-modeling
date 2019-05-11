use assembling::searching::banks::data_structure::int_graph::*;
use rdb2rdf::models::semantic_model::SemanticModel;
use fnv::FnvHashMap;
use std::prelude::v1::Vec;
use std::collections::HashSet;
use fnv::FnvHashSet;
use algorithm::data_structure::graph::*;
use std::collections::HashMap;
use algorithm::data_structure::graph::Node;
use permutohedron::heap_recursive;
use std::ops::AddAssign;


#[derive(Debug, Clone)]
pub struct EvalAttrMappingResult {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub gold_alignment: Vec<(String, String, String)>,
    pub pred_alignment: Vec<(String, String, String)>
}


pub fn evaluate_attr_mapping(int_graph: &IntGraph, sm: &SemanticModel, attr_mapping: &FnvHashMap<usize, usize>) -> EvalAttrMappingResult {
    // add attrs to gold attrs
    let mut gold_internal_nodes: FnvHashSet<usize> = Default::default();
    for (&attr_id, int_edge_id) in attr_mapping.iter() {
        let attr = sm.graph.get_node_by_id(attr_id);
        let edge= attr.first_incoming_edge(&sm.graph).unwrap();
        gold_internal_nodes.insert(edge.source_id);
    }

    let gold_numbered_internal_nodes = numbered_node_labels(&sm.graph, &gold_internal_nodes);
    let mut gold_attrs: HashSet<(usize, &str, &str)> = Default::default();

    for (&attr_id, int_edge_id) in attr_mapping.iter() {
        let attr = sm.graph.get_node_by_id(attr_id);
        let edge= attr.first_incoming_edge(&sm.graph).unwrap();
        gold_attrs.insert((attr_id, &edge.label, &gold_numbered_internal_nodes[&edge.source_id]));
    }

    // now we need loop through all possible permutation of internal class nodes, and select the best alignment
    let mut pred_internal_nodes_to_attrs: FnvHashMap<usize, Vec<usize>> = Default::default();
    for (&attr_id, &int_edge_id) in attr_mapping.iter() {
        let int_source_node = int_graph.graph.get_source_node(int_edge_id);
        pred_internal_nodes_to_attrs.entry(int_source_node.id).or_insert(Vec::new()).push(attr_id);
    }

    let mut pred_grouped_internal_nodes = group_nodes_by_label(&int_graph.graph, &mut pred_internal_nodes_to_attrs.keys());
    let mut pred_numbered_internal_nodes: FnvHashMap<usize, String> = Default::default();
    let mut pred_attrs: HashSet<(usize, &str, &str)> = Default::default();

    for (&n_lbl, pred_internal_nodes)in pred_grouped_internal_nodes.iter() {
        // need to loop through all permutation of this pred_internal_nodes
        let mut index = (1..pred_internal_nodes.len() + 1).collect::<Vec<_>>();
        let mut best_score = -1;
        let mut best_alignment = Vec::new();

        heap_recursive(&mut index, |permutation| {
            let numbered_nodes = permutation.iter()
                .map(|idx| format!("{}{}", pred_internal_nodes[0].label, idx))
                .collect::<Vec<_>>();

            let mut sub_pred_attrs: HashSet<(usize, &str, &str)> = Default::default();

            for (i, n) in pred_internal_nodes.iter().enumerate() {
                for attr_id in &pred_internal_nodes_to_attrs[&n.id] {
                    let int_edge = int_graph.graph.get_edge_by_id(attr_mapping[attr_id]);
                    sub_pred_attrs.insert((*attr_id, &int_edge.label, &numbered_nodes[i]));
                }
            }

            let score = sub_pred_attrs.intersection(&gold_attrs).count() as i32;
            if score > best_score {
                best_score = score;
                best_alignment = permutation.to_vec();
            }
        });

        for (i, idx) in best_alignment.iter().enumerate() {
            pred_numbered_internal_nodes.insert(pred_internal_nodes[i].id, format!("{}{}", pred_internal_nodes[i].label, idx));
        }
    }

    for (&attr_id, &int_edge_id) in attr_mapping.iter() {
        let int_edge = int_graph.graph.get_edge_by_id(int_edge_id);
        pred_attrs.insert((attr_id, &int_edge.label, &pred_numbered_internal_nodes[&int_edge.source_id]));
    }

    let mut tp = 0;
    let mut fp = 0;
    let mut fneg = 0;

    for pred_attr in &pred_attrs {
        if gold_attrs.contains(pred_attr) {
            tp += 1;
        } else {
            fp += 1;
        }
    }

    for gold_attr in &gold_attrs {
        if !pred_attrs.contains(gold_attr) {
            fneg += 1;
        }
    }

    let precision = tp as f32 / (tp + fp) as f32;
    let recall = tp as f32 / (tp + fneg) as f32;
    let f1 = if precision + recall == 0.0 {
        0.0 
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    let gold_alignment = gold_attrs.iter()
        .map(|&(attr_id, edge_lbl, source_lbl)| {
            (
                sm.graph.get_node_by_id(attr_id).label.clone(),
                edge_lbl.to_owned(),
                source_lbl.to_owned()
            )
        })
        .collect::<Vec<_>>();

    let pred_alignment = pred_attrs.iter()
        .map(|&(attr_id, edge_lbl, source_lbl)| {
            (
                sm.graph.get_node_by_id(attr_id).label.clone(),
                edge_lbl.to_owned(),
                source_lbl.to_owned()
            )
        })
        .collect::<Vec<_>>();

    EvalAttrMappingResult {
        accuracy: tp as f32 / (tp + fp + fneg) as f32,
        precision,
        recall,
        f1,
        gold_alignment,
        pred_alignment,
    }
}


pub fn numbered_node_labels(g: &Graph, nodes: &FnvHashSet<usize>) -> FnvHashMap<usize, String> {
    let mut numbered_labels: HashMap<&str, usize> = Default::default();
    let mut result: FnvHashMap<usize, String> = Default::default();

    for &nid in nodes {
        let n = g.get_node_by_id(nid);
        numbered_labels.entry(&n.label).or_insert(0).add_assign(1);
        result.insert(nid, format!("{}{}", n.label, numbered_labels[n.label.as_str()]));
    }

    result
}

pub fn group_nodes_by_label<'a, ND: NodeData, ED: EdgeData>(g: &'a Graph<ND, ED>, nodes: &mut Iterator<Item=&usize>) -> HashMap<&'a String, Vec<&'a Node<ND>>> {
    let mut groups: HashMap<&String, Vec<&Node<ND>>> = Default::default();
    for &nid in nodes {
        let n = g.get_node_by_id(nid);
        if !groups.contains_key(&n.label) {
            groups.insert(&n.label, Vec::new());
        }
        groups.get_mut(&n.label).unwrap().push(n);
    }

    groups
}