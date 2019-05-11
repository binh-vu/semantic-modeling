pub mod tests {
    use serde_json;
    use assembling::annotator::Annotator;
    use assembling::features::*;
    pub use std::io::*;
    pub use std::path::*;
    pub use std::fs::*;
    pub use rdb2rdf::prelude::*;
    pub use algorithm::prelude::*;
    use serde::Serialize;
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct RustInput {
        pub dataset: String,
        pub workdir: String,
        // place where you can freely write any data to
        pub semantic_models: Vec<SemanticModel>,
        pub train_sm_idxs: Vec<usize>,
        pub test_sm_idxs: Vec<usize>,

        #[serde(rename = "feature_primary_keys")]
        pub primary_keys: PrimaryKey,
        #[serde(rename = "feature_cardinality_features")]
        pub cardinality_features: CardinalityFeatures,
        #[serde(rename = "predicted_parent_stypes")]
        pub stype_assistant: STypeAssistant,
        pub ont_graph: OntGraph,
    }

    impl RustInput {
        pub fn get_annotator(&self) -> Annotator {
            let workdir = Path::new(&self.workdir);
            Annotator::new(
                &self.dataset, &workdir,
                &self.semantic_models, &self.train_sm_idxs,
                &self.stype_assistant,
                self.primary_keys.clone(), self.cardinality_features.clone())
        }

        pub fn get_input(file: &str) -> RustInput {
            let mut package_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let input_file = package_dir.join(file);
            serde_json::from_reader(BufReader::new(File::open(input_file).unwrap())).unwrap()
        }

        pub fn get_train_sms(&self) -> Vec<&SemanticModel> {
            self.train_sm_idxs.iter()
                .map(|&i| &self.semantic_models[i])
                .collect::<Vec<_>>()
        }
    }

    pub fn serialize_json<S: Serialize>(obj: &S, output: &str) {
        let mut package_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let input_file = package_dir.join(output);
        serde_json::to_writer_pretty(BufWriter::new(File::create(input_file).unwrap()), obj).unwrap();
    }

    pub fn deserialize_json<D>(input: &str) -> D
        where for<'de> D: Deserialize<'de> {
        let mut package_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let input_file = package_dir.join(input);
        serde_json::from_reader(BufReader::new(File::open(input_file).unwrap())).unwrap()
    }
}