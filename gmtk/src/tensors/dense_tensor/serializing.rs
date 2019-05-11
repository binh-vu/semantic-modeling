use std;
use serde::ser::{Serialize, Serializer, SerializeStruct};
use tensors::tensor_type::*;
use libc::c_void;
use tensors::DenseTensor;
use super::c_api::*;
use serde::Deserialize;
use std::marker::PhantomData;
use serde::de::Visitor;
use serde::Deserializer;
use std::fmt;
use serde::de;

impl<T: TensorType> Serialize for DenseTensor<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {

        let data = unsafe {
            let _tensor1 = cten_contiguous(self.tensor);
            let _tensor = cten_view1(_tensor1);
            cten_drop_tensor(_tensor1);
            let ptr = cten_get_data_ptr(_tensor) as *const <T as TensorType>::PrimitiveType;
            std::slice::from_raw_parts(ptr, cten_numel(self.tensor) as usize)
        };

        let dtype = unsafe { cten_get_dtype(self.tensor) };
        let shape: &[i64] = self.size();

        let mut state = serializer.serialize_struct("DenseTensor", 3)?;
        state.serialize_field("dtype", &dtype)?;
        state.serialize_field("shape", &shape)?;
        state.serialize_field("data", data)?;
        state.end()
    }
}

impl<'de, T: TensorType> Deserialize<'de> for DenseTensor<T> 
{
    fn deserialize<D>(deserializer: D) -> Result<DenseTensor<T>, D::Error>
    where D: Deserializer<'de> {
        struct TensorVisitor<T: TensorType> {
            _marker: std::marker::PhantomData<*const T>
        }

        impl<'de, T: TensorType> Visitor<'de> for TensorVisitor<T> 
        {
            type Value = DenseTensor<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct DenseTensor")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<DenseTensor<T>, V::Error>
                where
                    V: de::SeqAccess<'de>,
            {
                let dtype: i32 = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?;
                assert_eq!(dtype, T::get_dtype());
                let shape: Vec<i64> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let aten = unsafe {
                    match dtype {
                        3  => {
                            let data: Vec<i32> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(2, &self))?;
                            cten_from_array_and_view(data.len() as i64, shape.len(), shape.as_ptr(), data.as_ptr() as *const c_void, T::get_dtype(), DEFAULT_BACKEND)
                        },
                        4  => {
                            let data: Vec<i64> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(2, &self))?;
                            cten_from_array_and_view(data.len() as i64, shape.len(), shape.as_ptr(), data.as_ptr() as *const c_void, T::get_dtype(), DEFAULT_BACKEND)
                        },
                        5 => {
                            let data: Vec<f32> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(2, &self))?;
                            cten_from_array_and_view(data.len() as i64, shape.len(), shape.as_ptr(), data.as_ptr() as *const c_void, T::get_dtype(), DEFAULT_BACKEND)
                        },
                        6 => {
                            let data: Vec<f64> = seq.next_element()?.ok_or_else(|| de::Error::invalid_length(2, &self))?;
                            cten_from_array_and_view(data.len() as i64, shape.len(), shape.as_ptr(), data.as_ptr() as *const c_void, T::get_dtype(), DEFAULT_BACKEND)                            
                        },
                        _ => unimplemented!()
                    }
                };

                Ok(DenseTensor::<T>::new(aten))
            }
        }

        const FIELDS: &'static [&'static str] = &["dtype", "shape", "tensor"];
        deserializer.deserialize_struct("DenseTensor", FIELDS, TensorVisitor { _marker: PhantomData })
    }
}