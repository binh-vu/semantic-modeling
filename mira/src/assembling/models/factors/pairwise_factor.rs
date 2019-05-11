use gmtk::prelude::*;
use super::super::variable::*;

pub struct PairwiseFactor<'a> {
    pub(super) weights: &'a Weights,
    pub(super) vars: Vec<&'a TripleVar>,
    // represent vars_dims + weight.dim, e.g: [1, 2, 1, 2, 5]
    // where we have: 4 variables, input_var_idx = [1, 3], and |weight| = 5
    vars_dims_with_weight: Vec<i64>,
    sub_vars_dims: Vec<i64>,
    input_var_idx: Vec<usize>,
    pub(super) features_tensor: DenseTensor
}

impl<'a> PairwiseFactor<'a> {
    // Take list of variables belongs to a class node (incoming links and outgoing links) as `variables`,
    // and a list of variable index `input_var_idx` point to index of two variables that are actual input to the factor
    
}