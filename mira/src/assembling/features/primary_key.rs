use std::collections::HashMap;


#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct PrimaryKey {
    pesudo_primary_keys: HashMap<String, String>
}

impl Default for PrimaryKey {
    fn default() -> Self {
        PrimaryKey {
            pesudo_primary_keys: Default::default()
        }
    }
}

impl PrimaryKey {
    pub fn contains(&self, class_uri: &str) -> bool {
        self.pesudo_primary_keys.contains_key(class_uri)
    }

    pub fn get_primary_key(&self, class_uri: &str) -> &str {
        &self.pesudo_primary_keys[class_uri]
    }
}

