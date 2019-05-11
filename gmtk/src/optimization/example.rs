use graph_models::traits::LabeledVariable;
use graph_models::traits::Factor;
use optimization::accumulators::*;
use tensors::*;
use graph_models::inferences::*;
use fnv::FnvHashMap;


/// https://stackoverflow.com/questions/32300132/why-cant-i-store-a-value-and-a-reference-to-that-value-in-the-same-struct
pub struct NLLExample<'a: 'a1, 'a1, V: 'a + LabeledVariable, T: 'static + TensorType=TDefault> {
    target_assignment: FnvHashMap<usize, V::Value>,
    factors: &'a1 [Box<Factor<'a, V, T> + 'a>],
    inference: Box<Inference<'a, V, T> + 'a1>
}

impl<'a: 'a1, 'a1, V: 'a + LabeledVariable, T: 'static + TensorType> NLLExample<'a, 'a1, V, T> {
    pub fn new(variables: &'a [V], factors: &'a1 [Box<Factor<'a, V, T> + 'a>], inference: Box<Inference<'a, V, T> + 'a1>) -> NLLExample<'a, 'a1, V, T> {
        let target_assignment: FnvHashMap<usize, V::Value> = variables.iter().map(|v| (v.get_id(), v.get_label_value().clone())).collect();
        NLLExample {
            factors,
            target_assignment,
            inference
        }
    }

    pub fn accumulate_value(&mut self, loss_value: &mut ValueAccumulator) {
        self.inference.reset_value();
        self.inference.infer();

        loss_value.accumulate(self.inference.log_z());
        for factor in self.factors {
            loss_value.accumulate(-factor.score_assignment(&self.target_assignment));
        }
    }

    pub fn accumulate_value_and_gradient(&mut self, loss_value: &mut ValueAccumulator, gradient_accum: &mut Tensor1Accumulator<i64, T>) {
        self.inference.reset_value();
        self.inference.infer();

        loss_value.accumulate(self.inference.log_z());
        for factor in self.factors {
            loss_value.accumulate(-factor.score_assignment(&self.target_assignment));
            for (weight_id, gradient) in factor.compute_gradients(&self.target_assignment, self.inference.as_ref()) {
                gradient_accum.accumulate_minus(&weight_id, &gradient);
            }
        }
    }
}

pub struct MAPExample<'a: 'a1, 'a1, V: 'a + LabeledVariable, T: 'static + TensorType=TDefault> {
    target_assignment: FnvHashMap<usize, V::Value>,
    inference: Box<Inference<'a, V, T> + 'a1>
}

impl<'a: 'a1, 'a1, V: 'static + LabeledVariable, T: 'static + TensorType> MAPExample<'a, 'a1, V, T> {
    pub fn new(variables: &'a [V], _factors: &'a1 [Box<Factor<'a, V, T> + 'a>], inference: Box<Inference<'a, V, T> + 'a1>) -> MAPExample<'a, 'a1, V, T> {
        let target_assignment: FnvHashMap<usize, V::Value> = variables.iter().map(|v| (v.get_id(), v.get_label_value().clone())).collect();
        MAPExample {
            target_assignment,
            inference
        }
    }

    pub fn get_map_assignment(&mut self) -> FnvHashMap<usize, V::Value> {
        self.inference.reset_value();
        self.inference.infer();
        return self.inference.map();
    }

    pub fn get_target_assignment(&self) -> &FnvHashMap<usize, V::Value> {
        &self.target_assignment
    }
}
