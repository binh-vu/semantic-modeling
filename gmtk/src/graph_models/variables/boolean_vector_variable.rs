use graph_models::domains::BinaryVectorValue;
use graph_models::domains::BooleanVectorDomain;
use graph_models::traits::Domain;
use graph_models::traits::Variable;
use tensors::*;

pub struct BooleanVectorVariable<'a, U: 'static + TensorType + Sized = TDefault> {
    domain: &'a BooleanVectorDomain<U>,
    value: BinaryVectorValue<U>,
}

impl<'a, U: TensorType> BooleanVectorVariable<'a, U> {
    pub fn new(domain: &'a BooleanVectorDomain<U>, value: BinaryVectorValue<U>) -> BooleanVectorVariable<U> {
        BooleanVectorVariable {
            domain,
            value
        }
    }
}

impl<'a, U: TensorType> Variable for BooleanVectorVariable<'a, U> {
    type Value = BinaryVectorValue<U>;

    fn get_id(&self) -> usize {
        (self as *const _) as usize
    }

    #[inline]
    fn get_domain_size(&self) -> i64 {
        2
    }

    #[inline]
    fn get_domain(&self) -> &Domain<Value=BinaryVectorValue<U>> {
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