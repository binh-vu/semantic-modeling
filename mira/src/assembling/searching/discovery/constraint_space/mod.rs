mod search_node_data;
mod search_args;
mod discover_next_state;
mod merge_plan;
mod int_sub_graph;

pub use self::search_node_data::*;
pub use self::search_args::*;
pub use self::discover_next_state::*;
pub use self::merge_plan::*;
pub use self::int_sub_graph::*;


#[cfg(test)]
mod tests {
    use serde_json;
    use assembling::searching::banks::data_structure::int_graph::IntGraph;
    pub use assembling::tests::tests::*;

    pub fn load_int_graph() -> IntGraph {
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        serde_json::from_reader(BufReader::new(File::open(dir.join("resources/assembling/searching/int_graph.json")).unwrap())).unwrap()
    }

    pub fn load_input() -> RustInput {
        let mut input_file = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        input_file.push("resources/assembling/searching/rust-input.json");
        serde_json::from_reader(BufReader::new(File::open(input_file).unwrap())).unwrap()
    }

    pub fn get_attribute<T: AsRef<str>>(id: usize, lbl: &str, stypes: &[T], score: f32) -> Attribute {
        Attribute {
            id,
            label: lbl.to_owned(),
            semantic_types: stypes.iter()
                .map(|stype| get_semantic_type(stype.as_ref(), score))
                .collect::<Vec<_>>()
        }
    }

    pub fn get_semantic_type(stype: &str, score: f32) -> SemanticType {
        let mut split = stype.split("--");
        SemanticType {
            class_uri: split.next().unwrap().to_owned(),
            predicate: split.next().unwrap().to_owned(),
            score
        }
    }
}