use std::path::Path;
use mira::prelude::*;
use algorithm::prelude::*;
use serde_json;
use std::fs::*;
use std::io::*;
use rdb2rdf::ontology::ont_graph::OntGraph;

#[derive(Deserialize)]
pub struct RustInput {
    pub dataset: String,
    pub workdir: String, // place where you can freely write any data to
    pub semantic_models: Vec<SemanticModel>,
    pub train_sm_idxs: Vec<usize>,
    pub test_sm_idxs: Vec<usize>,

    #[serde(rename = "feature_primary_keys")]
    pub primary_keys: PrimaryKey,
    #[serde(rename = "feature_cardinality_features")]
    pub cardinality_features: CardinalityFeatures,
    #[serde(rename = "predicted_parent_stypes")]
    pub stype_assistant: STypeAssistant,
    pub ont_graph: OntGraph
}

#[derive(Deserialize, Debug)]
pub struct Configuration {
    pub settings: Settings
}

impl RustInput {
    pub fn get_annotator<'a>(&'a self) -> Annotator<'a> {
        let workdir = Path::new(&self.workdir);

        Annotator::new(
            &self.dataset, &workdir,
            &self.semantic_models, &self.train_sm_idxs,
            &self.stype_assistant,
            self.primary_keys.clone(), self.cardinality_features.clone())
    }

    pub fn from_file(finput: &Path) -> RustInput {
        serde_json::from_reader(BufReader::new(File::open(finput).unwrap())).unwrap()
    }
    
    pub fn get_sm_by_id(&self, sm_id: &str) -> Option<&SemanticModel> {
        for sm in self.semantic_models.iter() {
            if sm.id == sm_id {
                return Some(sm);
            }
        }

        return None;
    }

    pub fn get_train_sms(&self) -> Vec<&SemanticModel> {
        self.train_sm_idxs.iter()
            .map(|&i| &self.semantic_models[i])
            .collect::<Vec<_>>()
    }

    pub fn get_test_sms(&self) -> Vec<&SemanticModel> {
        self.test_sm_idxs.iter()
            .map(|&i| &self.semantic_models[i])
            .collect::<Vec<_>>()
    }

    pub fn iter_train_sms(&self) -> IterContainer<Self> {
        IterContainer::new(&self.train_sm_idxs, &self)
    }

    pub fn iter_test_sms(&self) -> IterContainer<Self> {
        IterContainer::new(&self.test_sm_idxs, &self)
    }
}

impl IterableContainer for RustInput {
    type Element = SemanticModel;

    fn get_element(&self, idx: usize) -> &Self::Element {
        &self.semantic_models[self.train_sm_idxs[idx]]
    }
}