use rdb2rdf::models::semantic_model::SemanticType;
use std::collections::HashMap;

/// We use semantic type to help justify if class C (not data node) should link to class A or class B.
/// Score is a potential gain if switching to another class (for example: potential gain if C link to B instead of A, currently C link to A)
/// Notice that parent_stypes can be NULL, store in empty string won't be a problem, since if we don't have parent stype, this
/// stype_assistant won't be invoked anyway.

#[derive(Serialize, Deserialize, Clone)]
pub struct STypeDetail {
    stype: SemanticType,
    parent_stypes: Vec<SemanticType>
}


#[derive(Serialize, Deserialize, Clone, Default)]
pub struct STypeAssistant {
    // because size of vec is small, linear search is actually faster than hashing
    stype_details: Vec<HashMap<String, Vec<STypeDetail>>>
}

impl STypeAssistant {
    pub fn get_potential_gain(&self, sm_idx: usize, attr: &str, class_uri: &str, predicate: &str, parent_class_uri: &str, parent_predicate: &str) -> Option<f32> {
        match self.stype_details[sm_idx][attr]
            .iter()
            .find(|stype| stype.stype.class_uri == class_uri && stype.stype.predicate == predicate) {
            None => None,
            Some(stype) => {
                if stype.parent_stypes.len() == 1 {
                    None
                } else {
                    match (0..stype.parent_stypes.len())
                        .find(|&i| stype.parent_stypes[i].class_uri == parent_class_uri && stype.parent_stypes[i].predicate == parent_predicate) {
                        None => None,
                        Some(i) => {
                            if i > 0 {
                                Some(stype.parent_stypes[i].score - stype.parent_stypes[0].score)
                            } else {
                                Some(stype.parent_stypes[0].score - stype.parent_stypes[i].score)
                            }
                        }
                    }
                }
            }
        }
    }
}