use tensors::*;
use num_traits::*;
use std::hash::Hash;
use graph_models::traits::*;
use graph_models::domains::*;
use std::ops::AddAssign;
use std::hash::Hasher;


pub struct MultiContinuousVectorVariable<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType + Sized=TDefault> {
    domain: &'a BinaryVectorDomain<V, U>,
    value: BinaryVectorValue<U>,
}

impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> Variable for MultiContinuousVectorVariable<'a, V, U> {
    type Value = BinaryVectorValue<U>;

    fn get_id(&self) -> usize {
        (self as *const _) as usize
    }

    fn get_domain_size(&self) -> i64 {
        self.domain.numel() as i64
    }

    fn set_value(&mut self, val: <Self as Variable>::Value) -> &Self {
        self.value = val;
        return self;
    }

    fn get_value(&self) -> &<Self as Variable>::Value {
        return &self.value;
    }
}

impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> MultiContinuousVectorVariable<'a, V, U> {
    pub fn new(domain: &'a BinaryVectorDomain<V, U>, value: BinaryVectorValue<U>) -> MultiContinuousVectorVariable<'a, V, U> {
        MultiContinuousVectorVariable { domain, value }
    }
}

impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> Hash for MultiContinuousVectorVariable<'a, V, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize((self as *const _) as usize);
    }
}

impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> PartialEq for MultiContinuousVectorVariable<'a, V, U> {
    fn eq(&self, other: &MultiContinuousVectorVariable<'a, V, U>) -> bool {
        return (self as *const _) as usize == (other as *const _) as usize;
    }
}
impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> Eq for MultiContinuousVectorVariable<'a, V, U> {}

pub struct MultiContinuousVectorVariableBuilder<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType + Sized=TDefault> {
    domain: &'a BinaryVectorDomain<V, U>,
    active_index: Vec<i64>,
    active_values: Vec<U::PrimitiveType>
}

impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> MultiContinuousVectorVariableBuilder<'a, V, U> {
    fn create(&self) -> MultiContinuousVectorVariable<'a, V, U> {
        let tensor = DenseTensor::<U>::zeros(&vec![self.domain.numel() as i64]);
        if self.active_index.len() > 0 {
            tensor.at(&self.active_index).assign(DenseTensor::<U>::borrow_from_array(&self.active_values));
        }
        let value = BinaryVectorValue {
            tensor: tensor,
            idx: 0,
        };

        MultiContinuousVectorVariable::new(self.domain, value)
    }
}

impl<'a, V: 'static + Eq + Hash + Clone, U: 'static + TensorType> AddAssign<(V, f64)> for MultiContinuousVectorVariableBuilder<'a, V, U> {
    fn add_assign(&mut self, rhs: (V, f64)) {
        self.active_index.push(self.domain.get_category_index(&rhs.0) as i64);
        self.active_values.push(U::PrimitiveType::from_f64(rhs.1).unwrap());
    }
}
