use tensors::*;
use graph_models::*;
use optimization::accumulators::*;
use optimization::optim_traits::Optimizer;

pub struct BasicGradientDescent<'a, T: 'static + TensorType=TDefault> {
    pub parameters: Vec<&'a Weights<T>>,
    pub gradient_accum: Tensor1AccumulatorDict<i64, T>,
    pub loss_accum: ValueAccumulator,
    pub lr: f64,
}

impl<'a, T: TensorType> BasicGradientDescent<'a, T> {
    pub fn new(parameters: Vec<&'a Weights<T>>, lr: f64) -> BasicGradientDescent<'a, T> {
        let mut gradient_accum = Tensor1AccumulatorDict::new();
        for param in &parameters {
            gradient_accum.track_object(param.id, &param.get_value());
        }

        BasicGradientDescent {
            parameters,
            gradient_accum,
            lr,
            loss_accum: ValueAccumulator::new()
        }
    }
}

impl<'a, T: TensorType> Optimizer<T> for BasicGradientDescent<'a, T> {

    fn get_loss_and_gradient_accum(&mut self) -> (&mut ValueAccumulator, &mut Tensor1AccumulatorDict<i64, T>) {
        (&mut self.loss_accum, &mut self.gradient_accum)
    }

    fn get_loss_accum(&mut self) -> &mut ValueAccumulator {
        &mut self.loss_accum
    }

    fn zero_grad(&mut self) {
        self.gradient_accum.clear();
        self.loss_accum.clear()
    }

    fn average(&mut self, n: usize) {
        self.loss_accum.value = self.loss_accum.value / n as f64;
        for val in self.gradient_accum.tensors.values_mut() {
            *val /= n as f64;
        }
    }

    fn step(&mut self) {
        for param in &self.parameters {
            *param.get_value_mut() -= self.gradient_accum.get_value(&param.id) * self.lr;
        }
    }
}