use tensors::*;
use std::hash::Hash;
use graph_models::traits::*;
use graph_models::domains::*;

pub struct BinaryVectorVariable<'a, V: 'static + Eq + Hash + Clone + Sync, T: 'static + TensorType=TDefault> {
    domain: &'a BinaryVectorDomain<V, T>,
    value: BinaryVectorValue<T>,
}

impl<'a, V: Eq + Hash + Clone + Sync, U: TensorType> BinaryVectorVariable<'a, V, U> {
    pub fn set_value_by_category(&mut self, val: &V) -> &Self {
        self.value = self.domain.encode_value(val);
        return self;
    }
}

impl<'a, V: Eq + Hash + Clone + Sync, T: TensorType> Variable for BinaryVectorVariable<'a, V, T> {
    type Value = BinaryVectorValue<T>;

    #[inline]
    fn get_id(&self) -> usize {
        (self as *const _) as usize
    }

    #[inline]
    fn get_domain_size(&self) -> i64 {
        self.domain.numel() as i64
    }

    #[inline]
    fn get_domain(&self) -> &Domain<Value=Self::Value> {
        self.domain
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