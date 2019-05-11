use r2rml::commands::modeling::SetInternalLinkCmd;
use r2rml::commands::modeling::SetSemanticTypeCmd;
use serde_json::Value;
use serde::Deserialize;
use serde::Serialize;

enum Command {
    SetSemanticType(SetSemanticTypeCmd),
    SetInternalLink(SetInternalLinkCmd),
}

impl Command {

    pub fn deserialize(val: &Value) -> Command {
        match val["cmd"].as_str() {
            Some("SetSemanticType") => Command::SetSemanticType(SetSemanticTypeCmd::deserialize(val)),
            Some("SetInternalLink") => Command::SetInternalLink(SetInternalLinkCmd::deserialize(val)),
            _ => panic!("Invalid command")
        }
    }

//    pub fn serialize(val: &Value) -> Command {
//        match val["cmd"] {
//
//        }
//    }
}