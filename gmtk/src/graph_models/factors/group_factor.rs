use tensors::*;
use graph_models::traits::*;
use graph_models::inferences::*;
use std::collections::HashSet;
use graph_models::utils::get_variables_index;
use fnv::FnvHashMap;


pub trait SubTensorFactor<'a, V: 'a + Variable, T: 'static + TensorType=TDefault>: Sync + Send {
    fn score_assignment(&self, assignment: &FnvHashMap<usize, V::Value>) -> f64;
    fn get_scores_tensor(&self) -> DenseTensor<T>;
    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, V::Value>, prob_factor: &DenseTensor<T>) -> Vec<(i64, DenseTensor<T>)>;
    fn debug(&self, _debug_container: &DebugContainer) {
        unimplemented!()
    }
}

pub struct GroupTensorFactor<'a, V: 'static + Variable, T: TensorType=TDefault> {
    factors: Vec<Box<SubTensorFactor<'a, V, T> + 'a>>,
    variables: Vec<&'a V>,
    vars_dims: Vec<i64>,
    variables_set: HashSet<usize>
}


impl<'a, V: 'a + Variable, T: 'static + TensorType> GroupTensorFactor<'a, V, T> {
    pub fn new(variables: Vec<&'a V>, factors: Vec<Box<SubTensorFactor<'a, V, T> + 'a>>) -> GroupTensorFactor<'a, V, T> {
        let vars_dims: Vec<i64> = variables.iter().map(|v| v.get_domain_size()).collect();
        let variables_set = get_variables_index(&variables);

        GroupTensorFactor {
            factors,
            variables,
            vars_dims,
            variables_set
        }
    }
}

impl<'a, V: 'a + Variable, T: 'static + TensorType> Factor<'a, V, T> for GroupTensorFactor<'a, V, T> {
    fn get_variables(&self) -> &[&'a V] {
        &self.variables
    }

    fn debug(&self, debug_container: &DebugContainer) {
        for factor in &self.factors {
            factor.debug(debug_container);
        }
    }

    fn score_assignment<'b>(&self, assignment: &FnvHashMap<usize, V::Value>) -> f64 {
        self.factors.iter().map(|f| f.score_assignment(assignment)).sum()
    }

    fn get_scores_tensor(&self) -> DenseTensor<T> {
        if self.factors.len() == 1 {
            self.factors[0].get_scores_tensor().expand(&self.vars_dims).contiguous_().view1()
        } else {
            let mut tensor = DenseTensor::<T>::zeros(&self.vars_dims);
            for f in &self.factors {
                tensor += f.get_scores_tensor();
            }
            tensor.view1()
        }
    }

    fn compute_gradients(&self, target_assignment: &FnvHashMap<usize, V::Value>, inference: &Inference<'a, V, T>) -> Vec<(i64, DenseTensor<T>)> {
        let prob_factor = inference.log_prob_factor(self).exp();
        let mut gradients = Vec::with_capacity(self.factors.len());
        for f in &self.factors {
            gradients.append(&mut f.compute_gradients(target_assignment, &prob_factor));
        }

        return gradients;
    }

    fn touch(&self, var: &V) -> bool {
        self.variables_set.contains(&var.get_id())
    }
}