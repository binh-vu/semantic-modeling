use algorithm::data_structure::graph::*;
use evaluation_metrics::semantic_modeling::*;

use assembling::auto_label::MRRLabel;
use assembling::auto_label::alignment::{ align_graph_no_ambiguous, align_graph_one };

#[inline]
pub fn max_f1<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode, max_permutation: usize) -> Option<MRRLabel> {
    let mut edge2label = Vec::with_capacity(pred_sm.n_edges);
    let alignment = align_graph_one(gold_sm, pred_sm, data_node_mode, max_permutation);
    if alignment.is_none() {
        return None;
    }
    let (f1, precision, recall, bijection, x_triples) = alignment.unwrap();

    for e in pred_sm.iter_edges() {
        // TODO: fix me, it doesn't take into account data_node_mode, and tightly couple with alignment internal code
        let mapped_triple = Triple {
            source_id: bijection.prime2x[e.source_id],
            predicate: &e.label,
            target_id: bijection.to_x(e.target_id)
        };

        edge2label.push(x_triples.contains(&mapped_triple));
    }

    Some(MRRLabel { f1, edge2label, bijection, precision, recall })
}

#[inline]
pub fn max_f1_no_ambiguous<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode, max_permutation: usize) -> Option<MRRLabel> {
    let alignment = align_graph_no_ambiguous(gold_sm, pred_sm, data_node_mode, max_permutation);
    if alignment.is_none() {
        None
    } else {
        let mut edge2label = Vec::with_capacity(pred_sm.n_edges);
        let (f1, precision, recall, bijection, x_triples) = alignment.unwrap();
        
        for e in pred_sm.iter_edges() {
            // TODO: fix me, it doesn't take into account data_node_mode, and tightly couple with alignment internal code
            let mapped_triple = Triple {
                source_id: bijection.prime2x[e.source_id],
                predicate: &e.label,
                target_id: bijection.to_x(e.target_id)
            };

            edge2label.push(x_triples.contains(&mapped_triple));
        }

        Some(MRRLabel { f1, edge2label, bijection, precision, recall })
    }
}