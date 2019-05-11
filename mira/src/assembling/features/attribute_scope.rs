use ndarray::prelude::Array2;
use std::collections::HashMap;
use rdb2rdf::models::semantic_model::SemanticModel;
use rdb2rdf::models::relational_schema::PATH_DELIMITER;

#[derive(Clone)]
pub struct AttributeScope {
    attribute_same_scope_matrix: Vec<Array2<bool>>,
    attr2idx: Vec<HashMap<String, usize>>,
    does_sm_has_hierachy: Vec<bool>
}

impl Default for AttributeScope {
    fn default() -> Self {
        AttributeScope {
            attribute_same_scope_matrix: Default::default(),
            attr2idx: Default::default(),
            does_sm_has_hierachy: Default::default()
        }
    }
}

impl AttributeScope {
    pub fn new(sms: &[SemanticModel]) -> AttributeScope {
        let attribute_same_scope_matrix = sms.iter()
            .map(|sm| {
                let mut scope_matrix = Array2::default((sm.attrs.len(), sm.attrs.len()));
                let attr_scope = sm.attrs.iter()
                    .map(|attr| {
                        match attr.label.rfind(PATH_DELIMITER) {
                            None => "".to_owned(),
                            Some(x) => attr.label.chars().take(x).collect::<String>()
                        }
                    })
                    .collect::<Vec<_>>();

                for (i, attr_i) in sm.attrs.iter().enumerate() {
                    for (j, attr_j) in sm.attrs.iter().enumerate() {
                        scope_matrix[(i, j)] = attr_scope[i] == attr_scope[j];
                    }
                }

                scope_matrix
            })
            .collect::<Vec<_>>();

        let attr2idx = sms.iter()
            .map(|sm| {
                sm.attrs.iter()
                    .enumerate()
                    .map(|(i, a)| (a.label.clone(), i))
                    .collect()
            })
            .collect::<Vec<_>>();

        let does_sm_has_hierachy = sms.iter().enumerate()
            .map(|(i, sm)| {
                !(0..sm.attrs.len()).all(|j| attribute_same_scope_matrix[i][(0, j)])
            })
            .collect::<Vec<_>>();

        AttributeScope {
            attribute_same_scope_matrix,
            attr2idx,
            does_sm_has_hierachy
        }
    }

    pub fn is_same_scope(&self, sm_idx: usize, attr_x: &str, attr_y: &str) -> bool {
        let x = &self.attr2idx[sm_idx];
        return self.attribute_same_scope_matrix[sm_idx][(x[attr_x], x[attr_y])];
    }

    pub fn does_sm_has_hierachy(&self, sm_idx: usize) -> bool {
        self.does_sm_has_hierachy[sm_idx]
    }
}