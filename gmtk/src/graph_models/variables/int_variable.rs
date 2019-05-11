use graph_models::domains::IntDomain;
use graph_models::traits::*;

pub struct IntVariable {
    domain: IntDomain,
    value: i32
}

impl IntVariable {
    pub fn new(domain: IntDomain, val: i32) -> IntVariable {
        IntVariable {
            domain,
            value: val
        }
    }
}

impl Variable for IntVariable {
    type Value = i32;

    fn get_id(&self) -> usize {
        (self as *const _) as usize
    }

    fn get_domain_size(&self) -> i64 {
        self.domain.numel() as i64
    }

    fn get_domain(&self) -> &Domain<Value=<Self as Variable>::Value> {
        &self.domain
    }

    fn set_value(&mut self, val: <Self as Variable>::Value) {
        self.value = val;
    }

    fn get_value(&self) -> &<Self as Variable>::Value {
        &self.value
    }
}