use std::fmt;

pub mod conf_misc;
pub mod conf_mrf;
pub mod conf_learning;
pub mod conf_predicting;
pub mod conf_search;
pub mod conf_features;
pub mod conf_ranking;

use self::conf_misc::*;
use self::conf_mrf::*;
use self::conf_learning::*;
use self::conf_predicting::*;

#[derive(Clone, Deserialize)]
pub struct Settings {
    // General arguments
    pub manual_seed: u8,

    pub mrf: MRFConf,
    pub learning: LearningConf,
    pub predicting: PredictingConf,

    // misc conf
    pub misc_conf: ConfMisc
}

static mut SETTINGS: Option<Settings> = None;

impl Settings {
    pub fn default() -> Settings {
        Settings {
            manual_seed: 120,
            mrf: MRFConf::default(),
            learning: LearningConf::default(),
            predicting: PredictingConf::default(),
            misc_conf: ConfMisc::default()
        }
    }

    pub fn get_instance<'a>() -> &'a Settings {
        unsafe {
            if SETTINGS.is_none() {
                SETTINGS = Some(Settings::default());
            }
            SETTINGS.as_ref().unwrap()
        }
    }

    pub fn update_instance(settings: Settings) {
        unsafe {
            if SETTINGS.is_none() {
                SETTINGS = Some(settings);
            } else {
                let _self = SETTINGS.as_mut().unwrap();
                _self.manual_seed           = settings.manual_seed.clone();
                _self.mrf                   = settings.mrf.clone();
                _self.learning              = settings.learning.clone();
                _self.predicting            = settings.predicting.clone();
                _self.misc_conf             = settings.misc_conf.clone();
            }
        }
    }
}

impl fmt::Debug for Settings {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#"
****************************** SETTINGS ******************************
manual_seed: {}
mrf: {:?}
learning: {:?}
predicting: {:?}
misc_conf: {:?}
**********************************************************************
"#, 
    self.manual_seed,
    self.mrf,
    self.learning,
    self.predicting,
    self.misc_conf)
    }
}