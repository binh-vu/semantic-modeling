use ndarray::prelude::Array2;
use std::collections::HashMap;
use rdb2rdf::prelude::SemanticModel;
use itertools::Itertools;
use settings::Settings;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct CooccurrenceFeatures {
    // occurrence matrix is the matrix of between two predicates within a class node
    class_matrices: HashMap<String, Array2<f32>>,
    pred2idx: HashMap<String, usize>,
    idx2pred: Vec<String>,
    pub top_corelated_pred: HashMap<String, Vec<(String, String)>>,
    min_support: f32
}

impl CooccurrenceFeatures {
    // Occurrence between predicates of a class nodes
    // we do count co-occurrence between a predicate and itself, because we have filter out
    // duplication edges, we can get the total in diagonal line

    pub fn new(sms: &[&SemanticModel]) -> CooccurrenceFeatures {
        let min_support = Settings::get_instance().mrf.features.cooccurrence.min_support;

        let mut pred2idx: HashMap<String, usize> = Default::default();
        let mut idx2pred: Vec<String> = Default::default();
        let mut class_matrices: HashMap<String, Array2<f32>> = Default::default();
        let mut top_corelated_pred: HashMap<String, Vec<(String, String)>> = Default::default();

        for sm in sms.iter() {
            let g = &sm.graph;
            for n in g.iter_class_nodes() {
                for e in n.iter_outgoing_edges(&g) {
                    if !pred2idx.contains_key(&e.label) {
                        pred2idx.insert(e.label.clone(), idx2pred.len());
                        idx2pred.push(e.label.clone());
                    }
                }
            }
        }

        for sm in sms.iter() {
            let g = &sm.graph;
            for n in g.iter_class_nodes() {
                if !class_matrices.contains_key(&n.label) {
                    class_matrices.insert(n.label.clone(), Array2::zeros((pred2idx.len(), pred2idx.len())));
                }

                let matrix = class_matrices.get_mut(&n.label).unwrap();

                for e_i in n.iter_outgoing_edges(&g).unique_by(|e| &e.label) {
                    for e_j in n.iter_outgoing_edges(&g).unique_by(|e| &e.label) {
                        matrix[(pred2idx[&e_i.label], pred2idx[&e_j.label])] += 1.0;
                    }
                }
            }
        }

        for (class_uri, matrix) in &class_matrices {
            // select top occurrence with min_support
            let mut correlated_pred = Vec::new();
            for i in 0..pred2idx.len() {
                for j in (i+1)..pred2idx.len() {
                    let cooccur = matrix[(i, j)] / matrix[(i, i)].max(matrix[(j, j)]);
                    if cooccur >= min_support {
                        correlated_pred.push((cooccur, i, j));
                    }
                }
            }

            correlated_pred.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            top_corelated_pred.insert(
                class_uri.clone(), 
                correlated_pred
                    .iter()
                    .map(|&(score, i, j)| (idx2pred[i].clone(), idx2pred[j].clone()))
                    .collect()
            );
        }

        CooccurrenceFeatures {
            class_matrices,
            pred2idx,
            idx2pred,
            top_corelated_pred,
            min_support
        }
    }

    pub fn get_asym_occurrence_prob(&self, class_uri: &str, pred_a: &str, pred_b: &str) -> Option<f32> {
        // test if pred_a is likely to cooccur with pred_b
        // note that: it's better when pred_a is more popular than pred_b because let's say
        // pred_a occur like 5 times and pred_b occur like 1 times with pred_a
        // then prob would be: 1/5 w.r.t pred_a, and 1/1 w.r.t pred_b
        if !self.class_matrices.contains_key(class_uri) || !self.pred2idx.contains_key(pred_a) || !self.pred2idx.contains_key(pred_b) {
            return None
        }

        let matrix = &self.class_matrices[class_uri];
        let pred_a_idx = self.pred2idx[pred_a];
        let pred_b_idx = self.pred2idx[pred_b];
        
        Some(matrix[(pred_a_idx, pred_b_idx)] / matrix[(pred_a_idx, pred_a_idx)])
    }

    pub fn get_occurrence_prob(&self, class_uri: &str, pred_a: &str, pred_b: &str) -> Option<f32> {
        // test if pred_a and pred_b is likely to occur together
        if !self.class_matrices.contains_key(class_uri) || !self.pred2idx.contains_key(pred_a) || !self.pred2idx.contains_key(pred_b) {
            return None
        }

        let matrix = &self.class_matrices[class_uri];
        let pred_a_idx = self.pred2idx[pred_a];
        let pred_b_idx = self.pred2idx[pred_b];
        
        Some(matrix[(pred_a_idx, pred_b_idx)] / matrix[(pred_a_idx, pred_a_idx)].max(matrix[(pred_b_idx, pred_b_idx)]))
    }

    #[inline]
    pub fn is_smaller_index(&self, pred_a: &str, pred_b: &str) -> bool {
        return self.pred2idx[pred_a] < self.pred2idx[pred_b];
    }

    pub fn get_occurrence_prob_min_support(&self, class_uri: &str, pred_a: &str, pred_b: &str) -> Option<f32> {
        // test if pred_a and pred_b is likely to occur together
        if !self.class_matrices.contains_key(class_uri) || !self.pred2idx.contains_key(pred_a) || !self.pred2idx.contains_key(pred_b) {
            return None
        }

        let matrix = &self.class_matrices[class_uri];
        let pred_a_idx = self.pred2idx[pred_a];
        let pred_b_idx = self.pred2idx[pred_b];
        
        let prob = matrix[(pred_a_idx, pred_b_idx)] / matrix[(pred_a_idx, pred_a_idx)].max(matrix[(pred_b_idx, pred_b_idx)]);
        if prob < self.min_support {
            None
        } else {
            Some(prob)
        }
    }
}