use std::fmt;


#[derive(Clone, Deserialize, Serialize)]
pub struct ConfMisc {
}

impl ConfMisc {
    pub fn default() -> ConfMisc {
        ConfMisc {
        }
    }
}

impl fmt::Debug for ConfMisc {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, r#""#)
    }
}