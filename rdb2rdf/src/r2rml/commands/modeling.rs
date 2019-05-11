use models::relational_schema::AttrPath;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub struct SetSemanticTypeCmd {
    node_id: String,
    source_uri: String,
    predicate: String,
    target_id: AttrPath,
}


pub struct SetInternalLinkCmd {
    source_id: String,
    target_id: String,
    predicate: String,
    source_uri: String,
    target_uri: String
}


impl SetSemanticTypeCmd {

    pub fn deserialize(val: &Value) -> SetSemanticTypeCmd {
        SetSemanticTypeCmd { node_id: "".to_owned(), source_uri: "".to_owned(), predicate: "".to_owned(), target_id: AttrPath::new(vec![]) }
//        match val["semantic_type"].as_str() {
//            Some(ref x) => {
//                let vec = x.split("---").collect();
//
//                SetSemanticTypeCmd {
//                    node_id: vec[0],
//                    source_uri: vec[0].chars().take(vec[0].len() - 1).collect(),
//                    predicate: vec[1],
//                }
//            }
//        }
    }
}

impl SetInternalLinkCmd {

    pub fn deserialize(val: &Value) -> SetInternalLinkCmd {
        SetInternalLinkCmd {
            source_uri: "".to_owned(),
            target_id: "".to_owned(),
            target_uri: "".to_owned(),
            source_id: "".to_owned(),
            predicate: "".to_owned()
        }
    }
}