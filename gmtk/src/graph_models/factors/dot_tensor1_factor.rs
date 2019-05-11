use tensors::*;
use graph_models::traits::*;
use graph_models::weights::Weights;
use graph_models::inferences::*;
use fnv::FnvHashMap;

pub trait DotTensor1Factor<'a, V: 'a + Variable, T: 'static + TensorType=TDefault>: Factor<'a, V, T>
    where Self: Sized {

    #[inline]
    fn get_weights(&self) -> &Weights<T>;
    fn get_features_tensor(&self) -> &DenseTensor<T>;
    #[inline]
    fn get_vars_dims(&self) -> &[i64];

    fn val2feature_idx(&self, values: &[&V::Value]) -> i64 {
        // Note: you may want to override this function to improve speed
        let vars = self.get_variables();

        ravel_index(&(0..vars.len())
            .map(|i| vars[i].get_domain().get_index(values[i]) as i64)
            .collect(),
            self.get_vars_dims()
        )
    }

    fn impl_get_scores_tensor(&self) -> DenseTensor<T> {
        self.get_features_tensor().mv(&self.get_weights().get_value())
    }

    fn impl_score_assignment(&self, assignment: &FnvHashMap<usize, V::Value>) -> f64 {
        let values: Vec<&V::Value> = self.get_variables().iter().map(|&v| &assignment[&v.get_id()]).collect();
        self.get_features_tensor()
            .at(slice![self.val2feature_idx(&values) as i64, ;])
            .dot(&self.get_weights().get_value()).get_f64()
    }

    fn impl_compute_gradients(&self, target_assignment: &FnvHashMap<usize, V::Value>, inference: &Inference<'a, V, T>) -> Vec<(i64, DenseTensor<T>)> {
        let values: Vec<&V::Value> = self.get_variables().iter().map(|&v| &target_assignment[&v.get_id()]).collect();
        let feature_tensor = self.get_features_tensor();
        let target_idx = self.val2feature_idx(&values) as i64;
        let tensor = inference.log_prob_factor(self);
        vec![(self.get_weights().id.clone(), feature_tensor.at(slice![target_idx, ;]) -
            tensor.exp().view(&[1, -1]).mm(&feature_tensor).squeeze())]
    }
}