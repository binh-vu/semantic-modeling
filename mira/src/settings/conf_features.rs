use std::fmt;

#[derive(Deserialize, Clone, Debug)]
pub struct Cooccurrence {
    pub min_support: f32
}

#[derive(Deserialize, Clone)]
pub struct FeaturesConf {
    pub cooccurrence: Cooccurrence
}

impl FeaturesConf {
    pub fn default() -> FeaturesConf {
        FeaturesConf {
            cooccurrence: Cooccurrence { min_support: 0.8 }
        }
    }
}

impl fmt::Debug for FeaturesConf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
        cooccurrence: {:?}"#,
        self.cooccurrence)
    }
}

