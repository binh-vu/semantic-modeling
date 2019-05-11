use tensors::*;
use graph_models::traits::*;
use fnv::FnvHashMap;

mod brute_force;
mod belief_propagation;
pub use self::belief_propagation::BeliefPropagation;
pub use self::brute_force::BruteForce;

#[derive(PartialEq, Eq)]
pub enum InferProb {
    MAP,
    MARGINAL
}

pub trait Inference<'a, V: 'a + Variable, T: TensorType=TDefault>: Sync + Send {

    /// Reset all previous value to do inference again
    fn reset_value(&mut self);

    /// Perform inference
    fn infer(&mut self);

    /// Compute MAP solution, should call this method or logZ first
    fn map(&self) -> FnvHashMap<usize, V::Value>;

    /// Compute log-Z, should call this method or map first!!!
    fn log_z(&self) -> f64;

    /// Compute log-prob of a variable
    fn log_prob_var(&self, var: &V) -> DenseTensor<T>;

    /// Compute log-prob of all neighbor variables of a factor
    fn log_prob_factor(&self, factor: &Factor<'a, V, T>) -> DenseTensor<T>;
}