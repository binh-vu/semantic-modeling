use tensors::*;
use graph_models::*;
use std::f64;

pub struct EarlyStopping<T: TensorType=TDefault> {
    loss_history: Vec<f64>,
    param_history: Vec<Vec<Weights<T>>>,
    // number of epochs with no improvement after which training will be stopped.
    patience: usize,
    // minimum change in the monitored quantity to qualify as an improvement, 
    // i.e. an absolute change of less than min_delta, will count as no improvement.
    min_delta: f64,
    best: f64,
    wait: usize
}

impl<T: TensorType> EarlyStopping<T> {
    pub fn new(min_delta: f64, patience: usize) -> EarlyStopping<T> {
        EarlyStopping {
            loss_history: Vec::new(),
            param_history: Vec::new(),
            min_delta,
            patience,
            best: f64::INFINITY,
            wait: 0
        }
    }

    pub fn recent_loss_history(&self) -> &[f64] {
        &self.loss_history[(self.loss_history.len() - self.patience)..self.loss_history.len()]
    }

    // Check if we should stop the training, also record parameters to return best parameters
    pub fn should_stop_w_weight(&mut self, loss_val: f64, params: Vec<Weights<T>>) -> Option<Vec<Weights<T>>> {
        let should_stop = self.should_stop(loss_val);
        self.param_history.push(params);

        if should_stop {
            // return the best params
            debug_assert_eq!(
                self.loss_history[(self.loss_history.len() - self.patience)..self.loss_history.len()].len(), 
                self.param_history.len() - 1);

            let (idx, _loss) = self.loss_history[(self.loss_history.len() - self.patience)..self.loss_history.len()]
                .iter().enumerate()
                .min_by(|a, b| b.1.partial_cmp(&a.1).unwrap()).unwrap();
            Some(self.param_history.swap_remove(idx + 1))
        } else {
            if self.param_history.len() > self.patience {
                self.param_history.remove(0);
            }
            None
        }
    }

    // Check if we should stop the training
    pub fn should_stop(&mut self, loss_val: f64) -> bool {
        self.loss_history.push(loss_val);
        if loss_val + self.min_delta < self.best {
            self.best = loss_val;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                return true;
            }
        }

        return false;
    }
}