use optimization::accumulators::{ ValueAccumulator, Tensor1AccumulatorDict };
use tensors::TensorType;

pub trait Optimizer<T: TensorType> {
    fn get_loss_and_gradient_accum(&mut self) -> (&mut ValueAccumulator, &mut Tensor1AccumulatorDict<i64, T>);
    fn get_loss_accum(&mut self) -> &mut ValueAccumulator;

    fn zero_grad(&mut self);
    fn average(&mut self, n: usize);
    fn step(&mut self);
}