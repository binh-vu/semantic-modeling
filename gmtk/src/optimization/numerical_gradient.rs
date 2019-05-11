use graph_models::*;
use tensors::*;
use std::ops::{Deref};

/// http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
pub fn compute_approx_gradient<T: TensorType, F>(weights: &Weights<T>, mut func: F, eps: f64) -> DenseTensor<T>
    where F: FnMut() -> f64
{
    let mut gradient = DenseTensor::<T>::zeros_like(weights.get_value().deref());
    debug_assert_eq!(weights.get_value().ndim(), 1);

    let numel = weights.get_value().numel();
    for i in 0..numel {
        let original_val = weights.get_value().at(i).get_f64();
        weights.get_value_mut().assign(i, original_val + eps);
        let first_point = func();
        weights.get_value_mut().assign(i, original_val - eps);
        let second_point = func();
        weights.get_value_mut().assign(i, original_val);
        gradient.assign(i, (first_point - second_point) / (2.0 * eps));
    }

    gradient
}