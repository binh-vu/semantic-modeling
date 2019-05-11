use tensors::*;
use graph_models::weights::Weights;
use fnv::FnvHashMap;
use graph_models::inferences::*;
use std::any::Any;

pub trait Domain {
    type Value;

    fn numel(&self) -> usize;
    fn get_index(&self, value: &Self::Value) -> usize;
    fn get_value(&self, index: usize) -> Self::Value;
}

pub trait Variable: Sync + Send {
    type Value: Clone + Sync + Send;

    /// Return a unique id, which will be used to hash variables, it must be unique in one example
    fn get_id(&self) -> usize;
    fn get_domain_size(&self) -> i64;
    fn get_domain(&self) -> &Domain<Value=Self::Value>;
    fn set_value(&mut self, val: Self::Value);
    fn get_value(&self) -> &Self::Value;
}

pub trait LabeledVariable: Variable {
    fn get_label_value(&self) -> &Self::Value;
}

pub trait DebugContainer {
    fn as_any(&self) -> &Any;
}

pub trait Factor<'a, V: 'a + Variable, T: 'static + TensorType=TDefault>: Sync + Send {
    fn get_id(&self) -> usize {
        ((self as *const _) as *const usize) as usize
    }

    fn get_variables(&self) -> &[&'a V];
    fn score_assignment(&self, assignment: &FnvHashMap<usize, V::Value>) -> f64;
    fn get_scores_tensor(&self) -> DenseTensor<T>;
    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, V::Value>, inference: &Inference<'a, V, T>) -> Vec<(i64, DenseTensor<T>)>;
    fn touch(&self, var: &V)-> bool;
    
    /// This function is solely for debugging purpose
    fn debug(&self, _debug_container: &DebugContainer) {
        unimplemented!()
    }
}

pub trait FactorTemplate<E, V: Variable, T: TensorType=TDefault>: Sync + Send {
    fn get_weights(&self) -> &[Weights<T>];
    fn unroll<'a: 'a1, 'a1>(&'a self, example: &'a1 E) -> Vec<Box<Factor<'a1, V, T> + 'a1>>;
}