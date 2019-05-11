use num_traits::cast::FromPrimitive;
use num_traits::cast::ToPrimitive;
use num_traits::cast::NumCast;
use num_traits::Num;
use std::fmt::Debug;
use serde::ser::Serialize;
use std::iter::Sum;

#[derive(Debug)]
#[repr(C)] pub enum ScalarType {
    Char = 0,
    Short = 1,
    Int = 2,
    Long = 3,
    Half = 4,
    Float = 5,
    Double = 6
}

#[derive(Debug)]
#[repr(C)] pub enum Backend {
    CPU = 0,
    CUDA = 1
}

pub trait TensorType: Debug {
    type PrimitiveType: Num + NumCast + FromPrimitive + ToPrimitive + Copy + Debug + Serialize + Sum;

    fn get_dtype() -> i32;
    fn get_scalar_type() -> ScalarType;
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TFloat;
impl TensorType for TFloat {
    type PrimitiveType = f32;

    fn get_dtype() -> i32 { ScalarType::Float as i32 }
    fn get_scalar_type() -> ScalarType { ScalarType::Float }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TDouble;
impl TensorType for TDouble {
    type PrimitiveType = f64;

    fn get_dtype() -> i32 { ScalarType::Double as i32 }
    fn get_scalar_type() -> ScalarType { ScalarType::Double }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TLong;
impl TensorType for TLong {
    type PrimitiveType = i64;

    fn get_dtype() -> i32 { ScalarType::Long as i32 }
    fn get_scalar_type() -> ScalarType { ScalarType::Long }
}

#[derive(Debug)]
pub struct TInt;
impl TensorType for TInt {
    type PrimitiveType = i32;

    fn get_dtype() -> i32 { ScalarType::Int as i32 }
    fn get_scalar_type() -> ScalarType { ScalarType::Int }
}

/// Specify default tensor type and default backend!
pub type TDefault = TFloat;
pub type TDScalar = f32;

pub static mut DEFAULT_BACKEND: i32 = Backend::CPU as i32;
pub fn set_default_backend(backend: Backend) {
    unsafe { DEFAULT_BACKEND = backend as i32; }
}