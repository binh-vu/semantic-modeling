#[derive(Deserialize, Clone, Debug)]
pub struct MicroRankingConf { 
    pub trigger_delta: f64, 
    pub coherence_weight: f64, 
    pub minimal_weight: f64 
}