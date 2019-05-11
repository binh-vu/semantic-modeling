use serde_json::{ Value, from_value };
use serde::Deserialize;
use serde::Deserializer;
use algorithm::data_structure::graph::Graph;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SemanticType {
    #[serde(rename = "domain")]
    pub class_uri: String,
    #[serde(rename = "type")]
    pub predicate: String,
    #[serde(rename = "confidence_score")]
    pub score: f32
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Attribute {
    pub id: usize,
    pub label: String,
    pub semantic_types: Vec<SemanticType>
}


#[derive(Serialize, Clone)]
pub struct SemanticModel {
    #[serde(rename = "name")]
    pub id: String,
    pub attrs: Vec<Attribute>,
    pub graph: Graph,
    #[serde(skip)]
    id2attrs: Vec<usize>
}

impl SemanticModel {
    pub fn new(id: String, attrs: Vec<Attribute>, graph: Graph) -> SemanticModel {
        let mut id2attrs: Vec<usize> = vec![attrs.len() + 100; graph.n_nodes];
        for (i, attr) in attrs.iter().enumerate() {
            id2attrs[attr.id] = i;
        }

        SemanticModel {
            id,
            attrs,
            graph,
            id2attrs
        }
    }

    pub fn empty(id: String) -> SemanticModel {
        SemanticModel {
            attrs: Vec::new(),
            graph: Graph::new(id.clone(), true, true, true),
            id2attrs: Vec::new(),
            id,
        }
    }

    pub fn get_attr_by_label(&self, lbl: &str) -> &Attribute {
        &self.attrs[self.id2attrs[self.graph.get_first_node_by_label(lbl).id]]
    }
}

impl<'de> Deserialize<'de> for SemanticModel {
    fn deserialize<D>(deserializer: D) -> Result<SemanticModel, D::Error>
        where D: Deserializer<'de> {

        let result = Value::deserialize(deserializer);
        match result {
            Err(e) => Err(e),
            Ok(mut val) => {
                let attrs: Vec<Attribute> = from_value(val["attrs"].take()).unwrap();
                let graph: Graph = Graph::from_dict(&val["graph"]);
                let mut id2attrs: Vec<usize> = vec![attrs.len() + 100; graph.n_nodes];

                for (i, attr) in attrs.iter().enumerate() {
                    id2attrs[attr.id] = i;
                }

                Ok(SemanticModel {
                    id: val["name"].as_str().unwrap().to_owned(),
                    attrs,
                    graph,
                    id2attrs
                })
            },
        }
    }
}