use tensors::*;
use graph_models::*;
use optimization::accumulators::*;
use optimization::optim_traits::Optimizer;
use std::ops::Deref;

/// COPIED FROM: https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html
/// 
/// Implements stochastic gradient descent (optionally with momentum).
/// 
/// Nesterov momentum is based on the formula from
/// `On the importance of initialization and momentum in deep learning`__.
/// 
/// Args:
///     params (iterable): iterable of parameters to optimize or dicts defining
///         parameter groups
///     lr (float): learning rate
///     momentum (float, optional): momentum factor (default: 0)
///     weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
///     dampening (float, optional): dampening for momentum (default: 0)
///     nesterov (bool, optional): enables Nesterov momentum (default: False)
/// 
/// Example:
///     >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
///     >>> optimizer.zero_grad()
///     >>> loss_fn(model(input), target).backward()
///     >>> optimizer.step()
/// 
/// __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
/// 
/// .. note::
/// The implementation of SGD with Momentum/Nesterov subtly differs from
/// Sutskever et. al. and implementations in some other frameworks.
/// 
/// Considering the specific case of Momentum, the update can be written as
/// 
/// .. math::
///           v = \rho * v + g \\
///           p = p - lr * v
/// 
/// where p, g, v and :math:`\rho` denote the parameters, gradient,
/// velocity, and momentum respectively.
/// 
/// This is in contrast to Sutskever et. al. and
/// other frameworks which employ an update of the form
/// 
/// .. math::
///      v = \rho * v + lr * g \\
///      p = p - v
/// 
/// The Nesterov version is analogously modified.
pub struct SGD<'a, T: 'static + TensorType=TDefault> {
    pub parameters: Vec<&'a Weights<T>>,
    pub gradient_accum: Tensor1AccumulatorDict<i64, T>,
    pub loss_accum: ValueAccumulator,
    pub lr: f64,
    pub momentum: f64,
    pub weight_decays: Vec<f64>,
    pub dampening: f64,
    pub nesterov: bool,
    pub groups: Vec<ParamState<T>>
}

pub struct SGDDefaultParams {
    pub momentum: f64,
    pub weight_decays: Vec<f64>,
    pub dampening: f64,
    pub nesterov: bool,
}

pub struct ParamState<T: TensorType> {
    momentum_buffer: DenseTensor<T>
}

impl<'a, T: TensorType> SGD<'a, T> {
    pub fn default_params() -> SGDDefaultParams {
        SGDDefaultParams {
            momentum: 0.0,
            weight_decays: vec![0.0],
            dampening: 0.0,
            nesterov: false
        }
    }

    pub fn new(parameters: Vec<&'a Weights<T>>, lr: f64, momentum: f64, mut weight_decays: Vec<f64>, dampening: f64, nesterov: bool) -> SGD<'a, T> {
        let mut gradient_accum = Tensor1AccumulatorDict::new();
        let mut groups = Vec::new();

        for param in &parameters {
            gradient_accum.track_object(param.id, &param.get_value());
            groups.push(ParamState {
                momentum_buffer: Default::default()
            });
        }

        if weight_decays.len() < parameters.len() {
            for i in weight_decays.len()..parameters.len() {
                weight_decays.push(0.0);
            }
        }

        SGD {
            parameters,
            gradient_accum,
            lr,
            loss_accum: ValueAccumulator::new(),
            momentum,
            weight_decays,
            dampening,
            nesterov,
            groups
        }
    }
}

impl<'a, T: TensorType> Optimizer<T> for SGD<'a, T> {

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
        for (i, param) in self.parameters.iter().enumerate() {
            let state = &mut self.groups[i];
            if self.weight_decays[i] != 0.0 {
                *self.gradient_accum.get_value_mut(&param.id) += param.get_value().deref() * self.weight_decays[i];
            }

            let d_p = self.gradient_accum.get_value(&param.id);
            let d_p = if self.momentum != 0.0 {
                if state.momentum_buffer.numel() == 0 {
                    state.momentum_buffer = DenseTensor::<T>::zeros_like(&param.get_value()) + d_p;
                } else {
                    state.momentum_buffer *= self.momentum;
                    state.momentum_buffer += (1.0 - self.dampening) * d_p;
                }

                if self.nesterov {
                    self.lr * (&state.momentum_buffer * self.momentum + d_p)
                } else {
                    self.lr * &state.momentum_buffer
                }
            } else {
                self.lr * d_p
            };

            *param.get_value_mut() -= d_p;
        }
    }
}