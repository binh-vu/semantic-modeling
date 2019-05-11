use graph_models::inferences::InferProb;
use graph_models::traits::*;
use tensors::*;
use rand::prelude::*;
use fnv::FnvHashMap;
use graph_models::inferences::belief_propagation::bp_structure::*;
use graph_models::inferences::belief_propagation::bp_dfs::DFSInfo;
use graph_models::inferences::Inference;

pub struct BeliefPropagation<'a: 'a2, 'a2, V: 'a + Variable, T: 'static + TensorType=TDefault> {
    pub(super) infer_prob: InferProb,
    pub(super) variables: Vec<BPVariable>,
    pub(super) factors: Vec<BPFactor<T>>,
    pub(super) edges: Vec<BPEdge<T>>,
    pub(super) ex_vars: &'a [V],
    pub(super) ex_factors: &'a2 [Box<Factor<'a, V, T> + 'a>],
    pub(super) ex_vars_index: FnvHashMap<usize, usize>,
    pub(super) ex_factors_index: FnvHashMap<usize, usize>,

    // vector of index of variables (not variable id)
    pub(super) roots: Vec<usize>,
    pub(super) roots_log_z: Vec<DenseTensor<T>>,
    // vector of component index of all nodes in factor graph (including var & factor), the component index start from 1
    // so that we can use usize
    pub(super) connected_components: Vec<usize>,
    pub(super) sending_plans: Vec<Vec<(usize, usize)>>,
    pub(super) has_cycle: bool,
    pub(super) is_inferred: bool
}

impl<'a: 'a2, 'a2, V: 'a + Variable, T: 'static + TensorType> BeliefPropagation<'a, 'a2, V, T> {
    pub fn new(infer_prob: InferProb, variables: &'a [V], factors: &'a2 [Box<Factor<'a, V, T> + 'a>], seed: u8) -> BeliefPropagation<'a, 'a2, V, T> {
        let n_edges: usize = factors.iter().map(|f| f.get_variables().len()).sum();
        let var_index: FnvHashMap<usize, usize> = variables.iter().enumerate().map(|(i, v)| (v.get_id(), i)).collect();
        let fac_index: FnvHashMap<usize, usize> = factors.iter().enumerate().map(|(i, f)| (f.get_id(), i)).collect();
        let _null_component: usize = variables.len() + factors.len() + 99000;

        let mut bp = BeliefPropagation {
            infer_prob,
            variables: Vec::with_capacity(variables.len()),
            factors: Vec::with_capacity(factors.len()),
            edges: Vec::with_capacity(n_edges),
            ex_vars: variables,
            ex_factors: factors,
            ex_vars_index: var_index,
            ex_factors_index: fac_index,
            roots: Vec::new(),
            roots_log_z: Vec::new(),
            connected_components: vec![_null_component; variables.len() + factors.len()],
            sending_plans: Vec::new(),
            has_cycle: false,
            is_inferred: false,
        };

        let mut id_counter = 0;

        // STEP 1: build factor graph: factor, variable & edges
        for factor in factors {
            bp.factors.push(BPFactor::new(id_counter, factor));
            id_counter += 1;
        }
        for var in variables {
            bp.variables.push(BPVariable::new(id_counter, var));
            id_counter += 1;
        }
        for (i, factor) in factors.iter().enumerate() {
            for var in factor.get_variables() {
                let var_idx = bp.ex_vars_index[&var.get_id()];
                bp.add_edge(i, var_idx);
            }
        }

        // STEP 2: need to find all disconnected components in the graph, because cannot pass message between those components
        let mut dfs = DFSInfo::new(bp.edges.len(), id_counter);
        let mut root_index = 0;

        let mut remained_variables = vec![0; variables.len()];
        let mut rng = StdRng::from_seed([seed; 32]);
        let mut random_idx = rng.gen_range(0, variables.len());

        loop {
            let vnode = &bp.variables[random_idx];
            bp.has_cycle = !dfs.dfs_from_var(&bp, -1, vnode, 0) || bp.has_cycle;
            // add roots, connected components, and make sending plans for this connected components
            bp.roots.push(random_idx);
            bp.roots_log_z.push(DenseTensor::<T>::default());

            let mut n_remained = 0;
            let mut n_connected_nodes = 0;
            for i in 0..factors.len() {
                if dfs.visit_order[i] != -1 {
                    bp.connected_components[i] = root_index;
                    n_connected_nodes += 1;
                }
            }

            for i in factors.len()..dfs.n_nodes {
                if dfs.visit_order[i] != -1 {
                    bp.connected_components[i] = root_index;
                    n_connected_nodes += 1;
                } else if bp.connected_components[i] == _null_component {
                    remained_variables[n_remained] = i - factors.len();
                    n_remained += 1;
                }
            }

            // add sending plan to send from leaves to root
            let mut sending_plan = Vec::with_capacity(n_connected_nodes - 1);
            for i in 0..dfs.n_nodes {
                if dfs.visit_order[i] > 0 { // 0 is root nodes
                    sending_plan.push((i, dfs.visit_order[i] as usize));
                }
            }
            sending_plan.sort_unstable_by(|&a, &b| a.1.cmp(&b.1).reverse());
            for i in 0..(n_connected_nodes - 1) {
                sending_plan[i].1 = dfs.from_edges[sending_plan[i].0] as usize;
            }
            bp.sending_plans.push(sending_plan);

            // randomly select another one variable to build, also reset tracker
            if n_remained == 0 {
                break;
            }
            random_idx = remained_variables[rng.gen_range(0, n_remained)];
            dfs.reset();
            root_index += 1;
        }

        bp
    }

    fn add_edge(&mut self, factor_idx: usize, var_idx: usize) {
        let edge = BPEdge::new(self.edges.len(), self.variables[var_idx].domain_size, var_idx, factor_idx);
        self.factors[factor_idx].edges.push(edge.id);
        self.variables[var_idx].edges.push(edge.id);
        self.edges.push(edge);
    }

    fn update_scores_tensor(&mut self) {
        for (i, factor) in self.ex_factors.iter().enumerate() {
            self.factors[i].score_tensor = factor.get_scores_tensor().view(&self.factors[i].dimensions);
        }
    }
}

impl<'a: 'a2, 'a2, V: 'a + Variable, T: 'static + TensorType> Inference<'a, V, T> for BeliefPropagation<'a, 'a2, V, T> {
    fn reset_value(&mut self) {
        if !self.is_inferred {
            return;
        }

        for edge in self.edges.iter_mut() {
            edge.reset_value();
        }
        for factor in self.factors.iter_mut() {
            factor.reset_value();
        }

        self.is_inferred = false;
    }

    fn infer(&mut self) {
        self.update_scores_tensor();

        for sending_plain in self.sending_plans.iter() {
            // sending from leaves to root
            for ue_pair in sending_plain.iter() {
                if ue_pair.0 < self.factors.len() {
                    // this is factor node
                    self.factors[ue_pair.0].send_message(&self.infer_prob, &mut self.edges, ue_pair.1);
                } else {
                    self.variables[ue_pair.0 - self.factors.len()].send_message(&mut self.edges, ue_pair.1);
                }
            }

            if self.infer_prob == InferProb::MARGINAL {
                for ue_pair in sending_plain.iter().rev() {
                    if ue_pair.0 < self.factors.len() {
                        self.variables[self.edges[ue_pair.1].var_idx].send_message(&mut self.edges, ue_pair.1);
                    } else {
                        self.factors[self.edges[ue_pair.1].factor_idx].send_message(&self.infer_prob, &mut self.edges, ue_pair.1);
                    }
                }
            }
        }

        for i in 0..self.roots.len() {
            self.roots_log_z[i] = self.variables[self.roots[i]].get_log_z(&self.edges);
        }

        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] at bp.rs");
        // // println!("[DEBUG] self.roots_log_z = {:?}", self.roots_log_z);
        // === [DEBUG] DEBUG CODE END   HERE ===
        

        self.is_inferred = true;
    }

    fn map(&self) -> FnvHashMap<usize, V::Value> {
        debug_assert!(self.is_inferred && self.infer_prob == InferProb::MAP);

        let mut encoded_solution = vec![0; self.variables.len()];
        let mut map_solution: FnvHashMap<usize, V::Value> = Default::default();

        for i in 0..self.roots.len() {
            self.variables[self.roots[i]].compute_map(self, &self.sending_plans[i], &mut encoded_solution);
        }

        for i in 0..self.variables.len() {
            map_solution.insert(self.ex_vars[i].get_id(), self.ex_vars[i].get_domain().get_value(encoded_solution[i] as usize));
        }

        map_solution
    }

    fn log_z(&self) -> f64 {
        debug_assert!(self.is_inferred && self.infer_prob == InferProb::MARGINAL);
        self.roots_log_z.iter().sum::<DenseTensor<T>>().get_f64()
    }

    fn log_prob_var(&self, var: &V) -> DenseTensor<T> {
        debug_assert!(self.is_inferred);

        let vidx = self.ex_vars_index[&var.get_id()];
        self.variables[vidx].get_log_belief(&self.edges) - &self.roots_log_z[self.connected_components[vidx + self.factors.len()]]
    }

    fn log_prob_factor(&self, factor: &Factor<'a, V, T>) -> DenseTensor<T> {
        debug_assert!(self.is_inferred);

        let fidx = self.ex_factors_index[&factor.get_id()];
        let result = self.factors[fidx].get_log_belief(&self.edges) - &self.roots_log_z[self.connected_components[fidx]];
        return result;
    }
}
