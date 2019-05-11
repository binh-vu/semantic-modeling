use gmtk::prelude::*;
use std::fmt;
use serde;
use serde::de::Deserializer;
use serde::de::{self, Visitor};
pub(super) static mut TRIPLE_VAR_DOMAIN: Option<BooleanVectorDomain> = None;

pub type TripleVarValue = BinaryVectorValue;
pub type TripleVarDomain = BooleanVectorDomain;
pub type TFDomain = BinaryVectorDomain<String>;

#[derive(Serialize, Deserialize, Clone)]
pub struct TripleVar {
    pub id: usize,
    #[serde(serialize_with = "serialize_triple_var_val", deserialize_with = "deserialize_triple_var_val")]
    value: TripleVarValue,
    #[serde(serialize_with = "serialize_triple_var_val", deserialize_with = "deserialize_triple_var_val")]
    labeled_value: TripleVarValue,
    pub edge_id: usize,
}

impl TripleVar {
    pub fn new(id: usize, value: TripleVarValue, edge_id: usize) -> TripleVar {
        TripleVar {
            labeled_value: value.clone(),
            id,
            value,
            edge_id
        }
    }
}

impl Variable for TripleVar {
    type Value = TripleVarValue;

    #[inline]
    fn get_id(&self) -> usize { self.id }

    #[inline]
    fn get_domain_size(&self) -> i64 {
        2
    }

    #[inline]
    fn get_domain(&self) -> &Domain<Value=<Self as Variable>::Value> {
        unsafe { TRIPLE_VAR_DOMAIN.as_ref().unwrap() }
    }

    #[inline]
    fn set_value(&mut self, val: <Self as Variable>::Value) {
        self.value = val;
    }

    #[inline]
    fn get_value(&self) -> &<Self as Variable>::Value {
        return &self.value;
    }
}

impl LabeledVariable for TripleVar {
    fn get_label_value(&self) -> &<Self as Variable>::Value {
        &self.labeled_value
    }
}

impl fmt::Debug for TripleVar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "TripleVar {{ id: {}, edge_id: {} }}", self.id, self.edge_id)
    }
}

fn serialize_triple_var_val<S>(x: &TripleVarValue, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer
{
    serializer.serialize_bool(x.idx == 1)
}

struct BoolVisitor;

impl<'de> Visitor<'de> for BoolVisitor {
    type Value = bool;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("bool value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<bool, E>
    where
        E: de::Error,
    {
        Ok(value)
    }
}


fn deserialize_triple_var_val<'de, D>(deserializer: D) -> Result<TripleVarValue, D::Error> 
    where D: Deserializer<'de> 
{
    let idx = deserializer.deserialize_bool(BoolVisitor)? as usize;
    let value = if idx == 0 {
        BinaryVectorValue {
            tensor: DenseTensor::from_array(&[1.0, 0.0]),
            idx: 0
        }
    } else {
        BinaryVectorValue {
            tensor: DenseTensor::from_array(&[0.0, 1.0]),
            idx: 1
        }
    };
    Ok(value)
}