use std::fmt;
use super::conf_search::*;
use super::conf_ranking::*;

#[derive(Clone, Deserialize, Debug)]
pub enum EarlyStopping {
    NoStop
}

#[derive(Clone, Deserialize, Debug)]
#[serde(tag="type")]
pub enum SearchMethod {
    BeamSearch(BeamSearchSettings)
}

#[derive(Clone, Deserialize, Debug)]
#[serde(tag="type")]
pub enum PostRanking {
    NoPostRanking,
    MicroRanking(MicroRankingConf)
}

#[derive(Deserialize, Clone)]
pub struct PredictingConf {
    pub search_method: SearchMethod,
    pub early_stopping: EarlyStopping,
    pub post_ranking: PostRanking,
}

impl PredictingConf {
    pub fn default() -> PredictingConf {
        PredictingConf {
            search_method: SearchMethod::BeamSearch(BeamSearchSettings { 
                beam_width: 10, 
                n_results: 10, 
                discovery: DiscoverMethod::ConstraintSpace(ConstraintSpace::default())
            }),
            early_stopping: EarlyStopping::NoStop,
            post_ranking: PostRanking::NoPostRanking
        }
    }
}

impl fmt::Debug for PredictingConf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
    search_method  : {:?}
    early_stopping : {:?}
    post_ranking   : {:?}"#, 
        self.search_method,
        self.early_stopping,
        self.post_ranking
        )
    }
}
