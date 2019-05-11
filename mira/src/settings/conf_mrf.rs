use std::fmt;
use super::conf_features::*;


#[derive(Deserialize, Clone)]
pub struct MRFConf {
    // graphical model arguments
    pub max_n_props: usize,
    // max n duplication per predicate, e.g: if a person have 2 cars, 3 phones, then 3 is max_n_duplications
    pub max_n_duplications: usize,
    // max predicate type can be duplicate (e.g: if person have 2 cars, 3 phones then max_n_duplication_types = 2)
    pub max_n_duplication_types: usize,

    // train args of MRF
    pub training_args: TrainingArgs,
    // features of MRF
    pub features: FeaturesConf,
    // templates
    pub templates: TemplatesConf,
}

impl MRFConf {
    pub fn default() -> MRFConf {
        MRFConf {
            max_n_props: 10,
            max_n_duplications: 5,
            max_n_duplication_types: 4,
            training_args: TrainingArgs::default(),
            features: FeaturesConf::default(),
            templates: TemplatesConf::default(),
        }
    }
}

impl fmt::Debug for MRFConf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
    max_n_props             : {}
    max_n_duplications      : {}
    max_n_duplication_types : {}
    templates: {:?}
    training_args: {:?}
    features: {:?}"#,
               self.max_n_props,
               self.max_n_duplications,
               self.max_n_duplication_types,
               self.templates,
               self.training_args,
               self.features)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct OptParams {
    pub lr: f64,
    pub eps: f64,
    // for adams only
    pub weight_decays: Vec<f64>, // L2 penalty
}

#[derive(Debug, PartialEq, Eq, Clone, Deserialize)]
pub enum OptimizerAlgo {
    BasicGradientDescent,
    Adam,
    SGD,
}

#[derive(Clone, Deserialize, Debug)]
pub struct EarlyStoppingArgs {
    pub patience: usize,
    pub min_delta: f64,
}

#[derive(Clone, Deserialize)]
pub struct TrainingArgs {
    pub n_epoch: usize,
    pub n_switch: usize,
    pub n_iter_eval: usize,
    pub mini_batch_size: usize,
    pub shuffle_mini_batch: bool,
    pub manual_seed: u64,
    pub report_final_loss: bool,
    pub optparams: OptParams,
    pub optimizer: OptimizerAlgo,
    pub parallel_training: bool,
    pub early_stopping: EarlyStoppingArgs,
}

impl TrainingArgs {
    pub fn default() -> TrainingArgs {
        TrainingArgs {
            n_epoch: 40,
            n_switch: 20,
            n_iter_eval: 5,
            mini_batch_size: 200,
            shuffle_mini_batch: false,
            manual_seed: 120,
            report_final_loss: true,
            optparams: OptParams {
                lr: 0.1,
                eps: 1e-8,
                weight_decays: vec![0.01],
            },
            early_stopping: EarlyStoppingArgs {
                patience: 5,
                min_delta: 0.001,
            },
            optimizer: OptimizerAlgo::Adam,
            parallel_training: false,
        }
    }
}

impl fmt::Debug for TrainingArgs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
        n_epoch            : {}                 
        n_switch           : {}                     
        n_iter_eval        : {}                     
        mini_batch_size    : {}                         
        shuffle_mini_batch : {}                             
        manual_seed        : {}                     
        report_final_loss  : {}                 
        optparams          : {:?}                     
        optimizer          : {:?}                     
        parallel_training  : {}
        early_stopping     : {:?}"#,
               self.n_epoch,
               self.n_switch,
               self.n_iter_eval,
               self.mini_batch_size,
               self.shuffle_mini_batch,
               self.manual_seed,
               self.report_final_loss,
               self.optparams,
               self.optimizer,
               self.parallel_training,
               self.early_stopping)
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct TemplatesConf {
    pub enable_duplication_factors: bool,
    pub enable_cooccurrence_factors: bool,
}

impl TemplatesConf {
    pub fn default() -> TemplatesConf {
        TemplatesConf {
            enable_duplication_factors: true,
            enable_cooccurrence_factors: true,
        }
    }
}

impl fmt::Debug for TemplatesConf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
        enable_duplication_factors : {}
        enable_cooccurrence_factors: {}"#,
               self.enable_duplication_factors,
               self.enable_cooccurrence_factors)
    }
}