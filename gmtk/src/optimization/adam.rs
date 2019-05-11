use tensors::*;
use graph_models::*;
use optimization::accumulators::*;
use optimization::optim_traits::Optimizer;

pub struct ParamState<T: TensorType> {
    step: i32,
    // Exponential moving average of gradient values
    exp_avg: DenseTensor<T>,
    // Exponential moving average of squared gradient values
    exp_avg_sq: DenseTensor<T>,
    // Maintains max of all exp. moving avg. of sq. grad. values
    max_exp_avg_sq: DenseTensor<T>
}

/// COPIED FROM: https://pytorch.org/docs/stable/_modules/torch/optim/adam.html
/// 
/// Implements Adam algorithm.
/// 
/// It has been proposed in `Adam: A Method for Stochastic Optimization`_.
/// 
/// Arguments:
///     params (iterable): iterable of parameters to optimize or dicts defining
///         parameter groups
///     lr (float, optional): learning rate (default: 1e-3)
///     betas (Tuple[float, float], optional): coefficients used for computing
///         running averages of gradient and its square (default: (0.9, 0.999))
///     eps (float, optional): term added to the denominator to improve
///         numerical stability (default: 1e-8)
///     weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
///     amsgrad (boolean, optional): whether to use the AMSGrad variant of this
///         algorithm from the paper `On the Convergence of Adam and Beyond`_
/// 
/// .. _Adam\: A Method for Stochastic Optimization:
///     https://arxiv.org/abs/1412.6980
/// .. _On the Convergence of Adam and Beyond:
///     https://openreview.net/forum?id=ryQu7f-RZ
pub struct Adam<'a, T: 'static + TensorType=TDefault> {
    pub parameters: Vec<&'a Weights<T>>,
    pub param_states: Vec<ParamState<T>>,
    pub gradient_accum: Tensor1AccumulatorDict<i64, T>,
    pub loss_accum: ValueAccumulator,
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decays: Vec<f64>,
    pub amsgrad: bool
}

pub struct DefaultAdamArgs {
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decays: Vec<f64>,
    pub amsgrad: bool
}

impl<'a, T: TensorType> Adam<'a,T> {
    pub fn default_params() -> DefaultAdamArgs {
        DefaultAdamArgs {
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decays: vec![0.0],
            amsgrad: false
        }
    }

    pub fn new(parameters: Vec<&'a Weights<T>>, lr: f64, betas: (f64, f64), eps: f64, mut weight_decays: Vec<f64>, amsgrad: bool) -> Adam<'a, T> {
        let mut gradient_accum = Tensor1AccumulatorDict::new();

        let mut param_states = Vec::with_capacity(parameters.len());
        for param in &parameters {
            gradient_accum.track_object(param.id, &param.get_value());
            param_states.push(ParamState {
                step: 0,
                exp_avg: DenseTensor::<T>::zeros_like(&param.get_value()),
                exp_avg_sq: DenseTensor::<T>::zeros_like(&param.get_value()),
                max_exp_avg_sq: DenseTensor::<T>::zeros_like(&param.get_value()),
            });
        }

        if weight_decays.len() < parameters.len() {
            for i in weight_decays.len()..parameters.len() {
                weight_decays.push(0.0);
            }
        }

        Adam {
            parameters,
            param_states,
            gradient_accum,
            lr,
            loss_accum: ValueAccumulator::new(),
            betas,
            eps,
            weight_decays,
            amsgrad
        }
    }
}

impl<'a, T: TensorType> Optimizer<T> for Adam<'a, T> {
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
            let state = &mut self.param_states[i];
            state.step += 1;

            let mut grad = self.gradient_accum.get_value(&param.id).clone_reference();
            if self.weight_decays[i] != 0.0 {
                grad = grad + &*param.get_value() * self.weight_decays[i];
            }

            // decay the first and second moment running average coefficient
            state.exp_avg *= self.betas.0;
            state.exp_avg += (1.0 - self.betas.0) * &grad;
            state.exp_avg_sq *= self.betas.1;
            state.exp_avg_sq.addcmul_(1.0 - self.betas.1, &grad, &grad);
            let mut denom = if self.amsgrad {
                // maintains the maximum of all 2nd moment running avg. till now
                state.max_exp_avg_sq = state.max_exp_avg_sq.max_w_tensor(&state.exp_avg_sq);
                // use the max. for normalizing running avg. of gradient
                state.max_exp_avg_sq.sqrt()
            } else {
                state.exp_avg_sq.sqrt()
            };
            denom += self.eps;
            let bias_correction1 = 1.0 - self.betas.0.powi(state.step);
            let bias_correction2 = 1.0 - self.betas.1.powi(state.step);
            let step_size = self.lr * bias_correction2.sqrt() / bias_correction1;

            param.get_value_mut().addcdiv_(-step_size, &state.exp_avg, &denom);
        }
    }
}