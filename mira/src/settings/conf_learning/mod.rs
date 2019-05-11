use std::fmt;

mod elimination;

pub use self::elimination::*;
use settings::conf_search::BeamSearchSettings;

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag="type")]
pub enum GenTrainDataMethod {
    IterativeMRRApproach { n_iter: usize, beam_settings: BeamSearchSettings, max_candidates_per_round: usize },
    GeneralBayesApproach { beam_settings: BeamSearchSettings, max_candidates_per_round: usize },
    TrialAndError { beam_width: usize, max_candidates_per_round: usize },
    Elimination(Elimination)
}

impl GenTrainDataMethod {
    pub fn get_name(&self) -> String {
        match self {
            GenTrainDataMethod::TrialAndError { .. } => "trial_and_error".to_owned(),
            GenTrainDataMethod::GeneralBayesApproach { .. } => "general_bayes_approach".to_owned(),
            GenTrainDataMethod::IterativeMRRApproach { n_iter, .. } => format!("iter_{}_mrr", n_iter),
            GenTrainDataMethod::Elimination(e) => "elimination".to_owned(),
        }
    }
}

#[derive(Clone, Deserialize, Debug)]
pub enum AutoLabelingMethod {
    MaxF1,
    PreservedStructure
}

impl Default for AutoLabelingMethod {
    fn default() -> AutoLabelingMethod {
        AutoLabelingMethod::MaxF1
    }
}

#[derive(Clone, Deserialize)]
pub struct LearningConf {
    pub max_n_examples: usize,
    pub max_permutation: usize,
    pub auto_labeling_method: AutoLabelingMethod,
    pub gen_data_method: GenTrainDataMethod
}

impl LearningConf {
    pub fn default() -> LearningConf {
        LearningConf {
            max_n_examples: 5000,
            max_permutation: 36000,
            auto_labeling_method: AutoLabelingMethod::MaxF1,
            gen_data_method: GenTrainDataMethod::TrialAndError {
                max_candidates_per_round: 30,
                beam_width: 10
            }
        }
    }
}

impl fmt::Debug for LearningConf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
    max_permutation: {}
    auto_labeling_method : {:?}
    gen_data_method: {:?}"#,
            self.max_permutation,
            self.auto_labeling_method,
            self.gen_data_method
        )
    }
}
