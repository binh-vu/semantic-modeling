use gmtk::graph_models::*;
use gmtk::tensors::*;
use assembling::models::variable::*;
use assembling::models::example::MRRExample;
use fnv::FnvHashMap;

pub struct TripleFactor<'a> {
    pub(super) weights: &'a Weights,
    pub(super) vars: [&'a TripleVar; 1],
    pub(super) vars_dims: [i64; 1],
    pub(super) features_tensor: DenseTensor
}

impl<'a> TripleFactor<'a> {
    pub fn new(var: &'a TripleVar, observed_features: &'a DenseTensor, weights: &'a Weights, domain_tensor: &'a DenseTensor) -> TripleFactor<'a> {
        let features_tensor = domain_tensor
            .matmul(&observed_features.view(&[1, 1, -1]))
            .view(&[2, -1]);

        TripleFactor {
            vars: [var],
            weights,
            features_tensor,
            vars_dims: [2]
        }
    }
}

impl<'a> Factor<'a, TripleVar> for TripleFactor<'a> {
    fn get_variables(&self) -> &[&'a TripleVar] {
        &self.vars
    }

    fn debug(&self, con: &DebugContainer) {
        self.debug_(con);
    }

    fn score_assignment(&self, assignment: &FnvHashMap<usize, <TripleVar as Variable>::Value>) -> f64 {
        self.impl_score_assignment(assignment)
    }

    fn get_scores_tensor(&self) -> DenseTensor {
        self.impl_get_scores_tensor()
    }

    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, <TripleVar as Variable>::Value>, inference: &Inference<'a, TripleVar>) -> Vec<(i64, DenseTensor)> {
        self.impl_compute_gradients(target_assignment, inference)
    }

    fn touch(&self, var: &TripleVar) -> bool {
        self.vars[0].id == var.id
    }
}

impl<'a> DotTensor1Factor<'a, TripleVar> for TripleFactor<'a> {
    #[inline]
    fn get_weights(&self) -> &Weights {
        self.weights
    }

    #[inline]
    fn get_features_tensor(&self) -> &DenseTensor {
        &self.features_tensor
    }

    #[inline]
    fn get_vars_dims(&self) -> &[i64] {
        &self.vars_dims
    }

    #[inline]
    fn val2feature_idx(&self, values: &[&<TripleVar as Variable>::Value]) -> i64 {
        values[0].idx as i64
    }
}