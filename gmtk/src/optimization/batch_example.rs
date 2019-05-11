use graph_models::traits::*;
use optimization::example::NLLExample;
use tensors::*;
use optimization::accumulators::*;
use rayon::prelude::*;

pub struct BatchNLLExample<'a: 'a1, 'a1: 'a2, 'a2, V: 'a + LabeledVariable, T: 'static + TensorType=TDefault> {
    pub examples: Vec<&'a2 mut NLLExample<'a, 'a1, V, T>>,
}

pub fn split_random<'a: 'a1, 'a1: 'a2, 'a2, V: 'a + LabeledVariable, T: 'static + TensorType>(examples: &'a2 mut [NLLExample<'a, 'a1, V, T>], batch_size: usize) -> Vec<Vec<&'a2 mut NLLExample<'a, 'a1, V, T>>> {
    let mut example_groups = Vec::new();
    for chunk in examples.chunks_mut(batch_size) {
        let mut group = Vec::new();
        for e in chunk {
            group.push(e);
        }
        example_groups.push(group);
    }


    example_groups
}

impl<'a: 'a1, 'a1: 'a2, 'a2, V: 'a + LabeledVariable, T: 'static + TensorType> BatchNLLExample<'a, 'a1, 'a2, V, T> {
    pub fn new(examples: &'a2 mut [NLLExample<'a, 'a1, V, T>]) -> BatchNLLExample<'a, 'a1, 'a2, V, T> {
        let mut batch_examples = Vec::new();
        for e in examples {
            batch_examples.push(e);
        }

        BatchNLLExample { examples: batch_examples }
    }

    pub fn accumulate_value(&mut self, loss_value: &mut ValueAccumulator) {
        for example in &mut self.examples {
            example.accumulate_value(loss_value);
        }        
    }

    pub fn accumulate_value_and_gradient(&mut self, loss_value: &mut ValueAccumulator, gradient_accum: &mut Tensor1AccumulatorDict<i64, T>) {
        for example in &mut self.examples {
            example.accumulate_value_and_gradient(loss_value, gradient_accum);
        }
    }

    pub fn size(&self) -> usize {
        self.examples.len()
    }
}

pub struct ParallelBatchNLLExample<'a: 'a1, 'a1: 'a2, 'a2, V: 'a + LabeledVariable, T: 'static + TensorType=TDefault> {
    pub examples: Vec<&'a2 mut NLLExample<'a, 'a1, V, T>>,
}

impl<'a: 'a1, 'a1: 'a2, 'a2, V: 'a + LabeledVariable, T: 'static + TensorType> ParallelBatchNLLExample<'a, 'a1, 'a2, V, T> {
    pub fn new(examples: &'a2 mut [NLLExample<'a, 'a1, V, T>]) -> ParallelBatchNLLExample<'a, 'a1, 'a2, V, T> {
        let mut batch_examples = Vec::new();
        for e in examples {
            batch_examples.push(e);
        }

        ParallelBatchNLLExample { examples: batch_examples }
    }

    pub fn accumulate_value(&mut self, loss_accum: &mut ValueAccumulator) {
        let loss_val: f64 = self.examples.par_iter_mut()
            .map(|ref mut e| {
                let mut laccum = ValueAccumulator::new();
                e.accumulate_value(&mut laccum);
                laccum.value
            })
            .sum();

        loss_accum.value = loss_val;
    }

    pub fn accumulate_value_and_gradient(&mut self, loss_accum: &mut ValueAccumulator, gradient_accum: &mut Tensor1AccumulatorDict<i64, T>) {
        let safe_grad_accum = SafeTensor1AccumulatorDict::new(gradient_accum);

        let loss_val: f64 = self.examples.par_iter_mut()
            .map(|ref mut e| {
                let mut laccum = ValueAccumulator::new();
                let mut gaccum = safe_grad_accum.clone_reference();
                e.accumulate_value_and_gradient(&mut laccum, &mut gaccum);

                laccum.value
            })
            .sum();

        loss_accum.value = loss_val;
        safe_grad_accum.update(gradient_accum);
    }

    pub fn size(&self) -> usize {
        self.examples.len()
    }
}
