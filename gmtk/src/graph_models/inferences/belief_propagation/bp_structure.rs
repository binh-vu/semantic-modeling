use graph_models::inferences::InferProb;
use graph_models::traits::*;
use tensors::*;
use graph_models::inferences::belief_propagation::bp::BeliefPropagation;

pub struct BPVariable {
    pub id: usize,
    pub domain_size: i64,
    pub edges: Vec<usize>
}
pub struct BPFactor<T: TensorType> {
    pub id: usize,
    pub edges: Vec<usize>,
    pub dimensions: Vec<i64>,
    pub score_tensor: DenseTensor<T>
}
pub struct BPEdge<T: TensorType=TDefault> {
    pub id: usize,
    pub var_idx: usize,
    pub factor_idx: usize,
    pub message2factor: DenseTensor<T>,
    pub message2var: DenseTensor<T>,
    pub trace_message: DenseTensor<TLong>
}

impl<T: 'static + TensorType> BPEdge<T> {
    pub fn new(id: usize, var_domain_size: i64, var_idx: usize, factor_idx: usize) -> BPEdge<T> {
        let dims = [var_domain_size];
        BPEdge {
            id,
            var_idx: var_idx,
            factor_idx: factor_idx,
            message2factor: DenseTensor::zeros(&dims),
            message2var: DenseTensor::zeros(&dims),
            trace_message: DenseTensor::zeros(&dims)
        }
    }

    pub fn reset_value(&mut self) {
        self.message2factor.zero_();
        self.message2var.zero_();
        self.trace_message.zero_();
    }
}

impl<T: 'static + TensorType> BPFactor<T> {
    pub fn new<'a, V: 'a + Variable>(id: usize, factor: &Box<Factor<'a, V, T> + 'a>) -> BPFactor<T> {
        BPFactor {
            id,
            edges: Vec::with_capacity(factor.get_variables().len()),
            dimensions: factor.get_variables().iter().map(|v| v.get_domain_size()).collect(),
            score_tensor: DenseTensor::default(),
        }
    }

    pub fn reset_value(&mut self) {
        self.score_tensor = DenseTensor::default()
    }

    pub fn send_message(&self, infer_prob: &InferProb, bp_edges: &mut Vec<BPEdge<T>>, eid: usize) {
        if self.edges.len() == 1 {
            bp_edges[eid].message2var = self.score_tensor.clone_reference();
            return;
        }

        let mut shape = vec![1; self.edges.len()];
        let mut along_dim = 0;
        let mut score_tensor = self.score_tensor.clone();

        for idx in 0..self.edges.len() {
            if self.edges[idx] == eid {
                along_dim = idx as i64;
                continue;
            }

            shape[idx] = -1;
            score_tensor += bp_edges[self.edges[idx]].message2factor.view(&shape);
            shape[idx] = 1;
        }

        let along_dim_size = score_tensor.size_in_dim(along_dim);
        if infer_prob == &InferProb::MAP {
            // unbind
            let (message, trace_messages) = score_tensor.unbind(along_dim).into_iter()
                .map(|mut aten| aten.contiguous_().view1().max_in_dim(0, true))
                .unzip();

            bp_edges[eid].message2var = DenseTensor::<T>::concat(&message, 0);
            bp_edges[eid].trace_message = DenseTensor::<TLong>::stack(&trace_messages, 0);
        } else {
            bp_edges[eid].message2var = score_tensor.swap_axes(0, along_dim).contiguous_().view(&[along_dim_size, -1]).log_sum_exp_2dim(1);
        }
    }

    #[inline]
    pub fn get_log_belief(&self, bp_edges: &Vec<BPEdge<T>>) -> DenseTensor<T> {
        // println!("[DEBUG] bp_structure.rs at 97");
        let mut shape = vec![1; self.edges.len()];
        shape[0] = -1;
        let mut score_tensor = &self.score_tensor + bp_edges[self.edges[0]].message2factor.view(&shape);
        shape[0] = 1;
        // println!("[DEBUG] bp_structure.rs at 102");
        for i in 1..self.edges.len() {
            shape[i] = -1;
            score_tensor += bp_edges[self.edges[i]].message2factor.view(&shape);
            shape[i] = 1;
        }
        // println!("[DEBUG] bp_structure.rs at 108");
        score_tensor
    }

    #[allow(dead_code)]
    pub fn get_log_prob(&self, bp_edges: &Vec<BPEdge<T>>) -> DenseTensor<T> {
        let mut log_belief = self.get_log_belief(bp_edges);
        log_belief -= log_belief.log_sum_exp();
        log_belief
    }
}

impl BPVariable {
    pub fn new<V: Variable>(id: usize, variable: &V) -> BPVariable {
        BPVariable {
            id,
            domain_size: variable.get_domain_size(),
            edges: Vec::new(),
        }
    }

    pub fn send_message<T: TensorType>(&self, bp_edges: &mut Vec<BPEdge<T>>, eid: usize) {
        // when len(edges) = 1 it's a leaf node, always 0 which is default message
        if self.edges.len() == 2 {
            bp_edges[eid].message2factor = if self.edges[0] == eid {
                bp_edges[self.edges[1]].message2var.clone_reference()
            } else {
                bp_edges[self.edges[0]].message2var.clone_reference()
            };
        } else {
            let mut message2factor = DenseTensor::<T>::zeros_like(&bp_edges[eid].message2var);
            for &e in &self.edges {
                if e != eid {
                    message2factor += &bp_edges[e].message2var;
                }
            }

            bp_edges[eid].message2factor = message2factor;
        }
    }

    pub fn compute_map<V: Variable, T: TensorType>(&self, bp: &BeliefPropagation<V, T>, sending_plan: &Vec<(usize, usize)>, solution: &mut [i64]) {
        let log_prob = self.get_log_belief::<T>(&bp.edges);
        let mut dimensions = [0; 50];

        let max_and_arg = log_prob.max_in_dim(0, false);
        solution[self.id - bp.factors.len()] = max_and_arg.1.get_i64();

        for ue_pair in sending_plan.iter().rev() {
            if ue_pair.0 < bp.factors.len() {
                // only back-prop through factor nodes to get argmax(y) of f(x, y)
                let edge = &bp.edges[ue_pair.1];
                let factor = &bp.factors[edge.factor_idx];

                if factor.edges.len() == 1 {
                    continue;
                }

                // unravel to set trace message correctly!!!
                let mut n_dim = 0;
                for &e in &factor.edges {
                    if e != ue_pair.1 {
                        dimensions[n_dim] = bp.variables[bp.edges[e].var_idx].domain_size;
                        n_dim += 1;
                    }
                }

                let idx = edge.trace_message.at(solution[edge.var_idx]).get_i64();
                let argmax = unravel_index_ptr(idx, &dimensions, n_dim);
                n_dim = 0;
                for &e in &factor.edges {
                    if e != ue_pair.1 {
                        solution[bp.edges[e].var_idx] = argmax[n_dim];
                        n_dim += 1;
                    }
                }
            }
        }
    }

    #[inline]
    pub fn get_log_z<T: TensorType>(&self, bp_edges: &Vec<BPEdge<T>>) -> DenseTensor<T> {
        self.get_log_belief(bp_edges).log_sum_exp()
    }

    #[inline]
    #[allow(dead_code)]
    pub fn get_log_prob<T: TensorType>(&self, bp_edges: &Vec<BPEdge<T>>) -> DenseTensor<T> {
        let mut log_belief = self.get_log_belief(bp_edges);
        log_belief -= &log_belief.log_sum_exp();
        log_belief
    }

    #[inline]
    pub fn get_log_belief<T: TensorType>(&self, bp_edges: &Vec<BPEdge<T>>) -> DenseTensor<T> {
        let mut log_prob = bp_edges[self.edges[0]].message2var.clone();
        for i in 1..self.edges.len() {
            log_prob += &bp_edges[self.edges[i]].message2var;
        }

        log_prob
    }
}