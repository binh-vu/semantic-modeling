use gmtk::prelude::*;
use fnv::FnvHashMap;
use super::super::variable::*;


#[derive(PartialEq, Eq, Debug)]
pub enum SufficientSubFactorType {
    DuplicationFactor,
    PairwisePKFactor,
    PairwiseScopeFactor,
    AllChildrenWrongFactor,
    PairwiseCooccurrenceFactor,
    STypeAssistantFactor
}

pub struct SufficientSubFactor<'a> {
    // this one is for debugging purpose
    pub(super) factor_type: SufficientSubFactorType,
    pub(super) weights: &'a Weights,
    pub(super) vars: Vec<&'a TripleVar>,
    // represent vars_dims + weight.dim, e.g: [1, 2, 1, 2, 5]
    // where we have: 4 variables, input_var_idx = [1, 3], and |weight| = 5
    vars_dims_with_weight: Vec<i64>,
    sub_vars_dims: Vec<i64>,
    pub(super) input_var_idx: Vec<usize>,
    pub(super) features_tensor: DenseTensor
}

impl<'a> SufficientSubFactor<'a> {
    // Take list of variables belongs to a class node (incoming links and outgoing links) as `variables`,
    // and a list of variable index `input_var_idx` point to index of two variables that are actual input to the factor

    pub fn new(factor_type: SufficientSubFactorType, vars: Vec<&'a TripleVar>, weights: &'a Weights, input_var_idx: Vec<usize>, features_tensor: DenseTensor) -> SufficientSubFactor<'a> {
        let mut vars_dims_with_weight = vec![1; vars.len() + 1];
        for &idx in &input_var_idx {
            vars_dims_with_weight[idx] = 2;
        }
        vars_dims_with_weight[vars.len()] = weights.get_value().size()[0];

        SufficientSubFactor {
            factor_type,
            weights,
            sub_vars_dims: vec![2; input_var_idx.len()],
            features_tensor: features_tensor.view(&[-1, vars_dims_with_weight[vars.len()]]),
            vars_dims_with_weight,
            vars,
            input_var_idx
        }
    }

    pub(super) fn assignment2feature_idx(&self, assignment: &FnvHashMap<usize, TripleVarValue>) -> i64 {
        ravel_index(&self.input_var_idx.iter()
            .map(|&idx| assignment[&self.vars[idx].get_id()].idx as i64)
            .collect::<Vec<_>>(),
            &self.sub_vars_dims
        )
    }
}

impl<'a> SubTensorFactor<'a, TripleVar> for SufficientSubFactor<'a> {
    fn score_assignment(&self, assignment: &FnvHashMap<usize, TripleVarValue>) -> f64 {
        return self.features_tensor
            .at(slice![self.assignment2feature_idx(assignment), ;])
            .dot(&self.weights.get_value()).get_f64();
    }

    fn get_scores_tensor(&self) -> DenseTensor {
        self.features_tensor.mv(&self.weights.get_value()).view(&self.vars_dims_with_weight[..self.vars_dims_with_weight.len() - 1])
    }

    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, TripleVarValue>, prob_factor: &DenseTensor) -> Vec<(i64, DenseTensor)> {
        let mut prob_factor_shape = prob_factor.size().to_owned();
        prob_factor_shape.push(1);

        let sub_prob_factor = prob_factor.view(&prob_factor_shape);
        let expectation = (self.features_tensor.view(&self.vars_dims_with_weight) * sub_prob_factor)
            .view(&[-1, self.vars_dims_with_weight[self.vars_dims_with_weight.len() - 1]])
            .sum_along_dim(0, false);
        let target_idx = self.assignment2feature_idx(target_assignment);

        return vec![(self.weights.id, self.features_tensor.at(target_idx) - expectation)]
    }

    fn debug(&self, _debug_container: &DebugContainer) {
        self.debug_(_debug_container);
    }
}

