use assembling::models::variable::*;
use gmtk::prelude::*;
use fnv::FnvHashMap;
use std::ops::AddAssign;

pub enum ExampleEnum<'a: 'a1, 'a1: 'a2, 'a2> {
    BatchNLL(BatchNLLExample<'a, 'a1, 'a2, TripleVar>),
    ParallelBatchNLL(ParallelBatchNLLExample<'a, 'a1, 'a2, TripleVar>),
}

impl<'a: 'a1, 'a1: 'a2, 'a2> ExampleEnum<'a, 'a1, 'a2> {
    #[allow(dead_code)]
    pub fn accumulate_value(&mut self, loss_accum: &mut ValueAccumulator) {
        match *self {
            ExampleEnum::BatchNLL(ref mut e) => e.accumulate_value(loss_accum),
            ExampleEnum::ParallelBatchNLL(ref mut e) => e.accumulate_value(loss_accum)
        }
    }

    pub fn accumulate_value_and_gradient(&mut self, loss_accum: &mut ValueAccumulator, gradient_accum: &mut Tensor1AccumulatorDict<i64, TDefault>) {
        match *self {
            ExampleEnum::BatchNLL(ref mut e) => e.accumulate_value_and_gradient(loss_accum, gradient_accum),
            ExampleEnum::ParallelBatchNLL(ref mut e) => e.accumulate_value_and_gradient(loss_accum, gradient_accum)
        }
    }

    pub fn size(&self) -> usize {
        match *self {
            ExampleEnum::BatchNLL(ref e) => e.size(),
            ExampleEnum::ParallelBatchNLL(ref e) => e.size()
        }
    }
}


fn get_confusion_matrix(assignment: &FnvHashMap<usize, TripleVarValue>, target_assignment: &FnvHashMap<usize, TripleVarValue>, domain: &TripleVarDomain) -> ConfusionMatrix {
    let matrix = ConfusionMatrix::create(domain.numel(), vec!["false".to_owned(), "true".to_owned()]);

    for (k, v) in target_assignment.iter() {
        let pred_v = &assignment[k];
        matrix.matrix.at((v.idx as i64, pred_v.idx as i64)).add_assign(1.0);
    }

    matrix
}

pub fn evaluate(map_examples: &mut Vec<MAPExample<TripleVar>>, domain: &TripleVarDomain) -> Option<ConfusionMatrix> {
    if map_examples.len() == 0 {
        None
    } else {
        Some(map_examples.iter_mut()
            .map(|e| get_confusion_matrix(&e.get_map_assignment(), e.get_target_assignment(), domain))
            .fold(ConfusionMatrix::create(2, vec!["false".to_owned(), "true".to_owned()]), |a, b| a + b))
    }
}