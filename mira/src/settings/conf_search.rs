#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct BeamSearchSettings {
    pub beam_width: usize,
    pub n_results: usize,
    pub discovery: DiscoverMethod
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct SMFilter {}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct CPMergePlanFilter {
    pub enable: bool,
    pub max_n_empty_hop: usize
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct ConstraintSpace {
    pub beam_width: usize,
    pub merge_plan_filter: CPMergePlanFilter
}

#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct GeneralDiscovery {
    pub beam_width: usize,
    pub max_class_node_hop: usize,
    pub max_data_node_hop: usize,
    pub triple_adviser_max_candidate: usize,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag="type")]
pub enum DiscoverMethod {
    GeneralDiscovery(GeneralDiscovery),
    ConstraintSpace(ConstraintSpace)
}

impl DiscoverMethod {
    pub fn as_general_discovery(&self) -> &GeneralDiscovery {
        if let DiscoverMethod::GeneralDiscovery(o) = &self {
            o
        } else {
            panic!("Cannot convert: {:?} to DiscoverMethod::GeneralDiscovery", self)
        }
    }
}

impl ConstraintSpace {
    pub fn default() -> ConstraintSpace {
        ConstraintSpace {
            beam_width: 10,
            merge_plan_filter: CPMergePlanFilter {
                enable: true,
                max_n_empty_hop: 2
            }
        }
    }
}