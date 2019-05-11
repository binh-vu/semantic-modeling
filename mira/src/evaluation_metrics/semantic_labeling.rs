// use std::iter::Sum;
use std::collections::HashMap;
use algorithm::prelude::Graph;
use rdb2rdf::prelude::*;

pub fn mrr(gold_sm: &SemanticModel, pred_sm: &SemanticModel) -> f64 {
    // The learned semantic types should be stored in pred_sm.attributes
    let mut ranks = vec![];
    for attr in &gold_sm.attrs {
        let pred_attr = pred_sm.get_attr_by_label(&attr.label);
        if !gold_sm.graph.has_node_with_id(attr.id) {
            // this column is ignored, and not used in sm
            if pred_attr.semantic_types.len() == 0 {
                ranks.push(1.0);
            } else {
                ranks.push(0.0)
            }
        } else {
            let node = gold_sm.graph.get_node_by_id(attr.id);
            assert_eq!(node.n_incoming_edges, 1, "Not support multi-parents");
            let edge = node.first_incoming_edge(&gold_sm.graph).unwrap();
            let gold_st_domain = &edge.get_source_node(&gold_sm.graph).label;

            let mut found_in_prediction = false;
            for (i, pred_st) in pred_attr.semantic_types.iter().enumerate() {
                if &pred_st.class_uri == gold_st_domain && pred_st.predicate == edge.label {
                    ranks.push(1.0 / (i + 1) as f64);
                    found_in_prediction = true;
                    break;
                }
            }
            if !found_in_prediction {
                ranks.push(0.0);
            }
        }
    }

    ranks.iter().sum::<f64>() / ranks.len() as f64
}

pub fn accuracy(gold_sm: &SemanticModel, pred_sm: &SemanticModel) -> f64 {
    // The learned semantic types should be stored in pred_sm.attributes
    accuracy_(&gold_sm.graph, &pred_sm.graph)
}

pub(super) fn accuracy_(gold_sm: &Graph, pred_sm: &Graph) -> f64 {
    // The learned semantic types should be stored in pred_sm.attributes
    let mut dnodes: HashMap<String, (String, String)> = Default::default();
    for dnode in gold_sm.iter_data_nodes() {
        let dlink = dnode.first_incoming_edge(&gold_sm).unwrap();
        dnodes.insert(dnode.label.clone(), (dlink.get_source_node(&gold_sm).label.clone(), dlink.label.clone()));
    }

    assert_eq!(dnodes.len(), gold_sm.iter_data_nodes().count(), "Label of data nodes must be unique");
    let mut prediction: HashMap<_, _> = Default::default();
    for dnode in pred_sm.iter_data_nodes() {
        let dlink = dnode.first_incoming_edge(&pred_sm).unwrap();
        let dstypes = &dnodes[&dnode.label];

        if &dstypes.0 == &dlink.get_source_node(&pred_sm).label && &dstypes.1 == &dlink.label {
            prediction.insert(dnode.label.clone(), 1.0);
        } else {
            prediction.insert(dnode.label.clone(), 0.0);
        }
    }

    for lbl in dnodes.keys() {
        if !prediction.contains_key(lbl) {
            prediction.insert(lbl.clone(), 0.0);
        }
    }

    assert_eq!(prediction.len(), dnodes.len());
    prediction.values().sum::<f64>() / prediction.len() as f64
}

pub fn coverage(gold_sm: &SemanticModel, pred_sm: &SemanticModel) -> f64 {
    // The learned semantic types should be stored in pred_sm.attributes
    let mut ranks = vec![];
    for attr in &gold_sm.attrs {
        let pred_attr = pred_sm.get_attr_by_label(&attr.label);
        if !gold_sm.graph.has_node_with_id(attr.id) {
            // this column is ignored, and not used in sm
            if pred_attr.semantic_types.len() == 0 {
                ranks.push(1.0);
            } else {
                ranks.push(0.0)
            }
        } else {
            let node = gold_sm.graph.get_node_by_id(attr.id);
            assert_eq!(node.n_incoming_edges, 1, "Not support multi-parents");
            let edge = node.first_incoming_edge(&gold_sm.graph).unwrap();
            let gold_st_domain = &edge.get_source_node(&gold_sm.graph).label;

            let mut found_in_prediction = false;
            for (i, pred_st) in pred_attr.semantic_types.iter().enumerate() {
                if &pred_st.class_uri == gold_st_domain && pred_st.predicate == edge.label {
                    ranks.push(1.0);
                    found_in_prediction = true;
                    break;
                }
            }
            if !found_in_prediction {
                ranks.push(0.0);
            }
        }
    }

    ranks.iter().sum::<f64>() / ranks.len() as f64
}