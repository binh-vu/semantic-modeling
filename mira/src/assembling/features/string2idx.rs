use rdb2rdf::models::semantic_model::SemanticModel;
use std::collections::HashMap;

pub struct String2Idx {
    string2idx: HashMap<String, u32>
}

impl String2Idx {
    pub fn new(train_sms: &[SemanticModel]) -> String2Idx {
        let mut string2idx: HashMap<String, u32> = Default::default();
        let mut counter = 0;
        for sm in train_sms {
            for n in sm.graph.iter_nodes() {
                if !string2idx.contains_key(&n.label) {
                    string2idx.insert(n.label.clone(), counter);
                    counter += 1;
                }
            }

            for e in sm.graph.iter_edges() {
                if !string2idx.contains_key(&e.label) {
                    string2idx.insert(e.label.clone(), counter);
                    counter += 1;
                }
            }
        }

        String2Idx {
            string2idx
        }
    }

    #[inline]
    pub fn idx(&self, s: &str) -> u32 {
        self.string2idx[s]
    }

    #[inline]
    pub fn get(&self, s: &str) -> Option<&u32> {
        self.string2idx.get(s)
    }

    #[inline]
    pub fn has(&self, s: &str) -> bool {
        self.string2idx.contains_key(s)
    }
}