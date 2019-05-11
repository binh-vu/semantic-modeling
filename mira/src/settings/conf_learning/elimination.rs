use settings::conf_search::DiscoverMethod;


#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct Elimination {
    pub discover_method: DiscoverMethod,
    pub n_elimination: usize,
    pub n_candidates: usize
}