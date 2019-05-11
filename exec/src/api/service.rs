use serde_json;
use super::in_mem_db::*;
use super::errors::*;

pub struct Command {
    name: String,
    args: serde_json::Value
}

pub fn execute_command_with_args(raw_cmd: &str, in_memory_db: &mut InMemoryDB) -> Result<serde_json::Value> {
//    let command: Command = serde_json::from_str(&raw_cmd).chain_err(|| "Invalid JSON")?;

//    match command.name.as_str() {
//        "register_input" => {
//        },
//        "generate_train_data" => {
//
//        },
//        _ => unreachable!()
//    }
    unimplemented!()
}