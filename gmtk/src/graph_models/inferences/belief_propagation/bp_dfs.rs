use graph_models::inferences::belief_propagation::bp_structure::*;
use tensors::TensorType;
use graph_models::inferences::belief_propagation::bp::BeliefPropagation;
use graph_models::traits::Variable;

pub struct DFSInfo {
    pub n_nodes: usize,
    pub n_edges: usize,

    // array of edge id which a node is visited from during DFS, -1 means no edges
    pub from_edges: Vec<i32>,
    // keep order of visiting, -1 mean that a node hasn't been visited
    pub visit_order: Vec<i32>,
    // keep information whether an edge has been visited or not (0 and 1)
    pub visited_edges: Vec<i32>
}

impl DFSInfo {
    pub fn new(n_edges: usize, n_nodes: usize) -> DFSInfo {
        DFSInfo {
            n_nodes,
            n_edges,
            from_edges: vec![-1; n_nodes],
            visit_order: vec![-1; n_nodes],
            visited_edges: vec![0; n_edges],
        }
    }

    /// Compute DFS travel path and return true if graph doesn't have cycle, false otherwise
    pub fn dfs_from_factor<V: Variable, T: TensorType>(&mut self, bp: &BeliefPropagation<V, T>, from_edge_id: i32, node: &BPFactor<T>, order: i32) -> bool {
        if self.visit_order[node.id] != -1 {
            // has cycle
            return false;
        }

        self.visit_order[node.id] = order;
        self.from_edges[node.id] = from_edge_id;
        let mut no_cycle = true;

        for &eid in &node.edges {
            if self.visited_edges[eid] != 0 {
                continue;
            }

            self.visited_edges[eid] = 1;
            no_cycle = self.dfs_from_var(bp, eid as i32, &bp.variables[bp.edges[eid].var_idx], order + 1) && no_cycle;
        }

        return no_cycle;
    }

    /// Compute DFS travel path and return true if graph doesn't have cycle, false otherwise
    pub fn dfs_from_var<V: Variable, T: TensorType>(&mut self, bp: &BeliefPropagation<V, T>, from_edge_id: i32, node: &BPVariable, order: i32) -> bool {
        if self.visit_order[node.id] != -1 {
            // has cycle
            return false;
        }

        self.visit_order[node.id] = order;
        self.from_edges[node.id] = from_edge_id;
        let mut no_cycle = true;

        for &eid in &node.edges {
            if self.visited_edges[eid] != 0 {
                // ignored visited path
                continue;
            }

            self.visited_edges[eid] = 1;
            no_cycle = self.dfs_from_factor(bp, eid as i32, &bp.factors[bp.edges[eid].factor_idx], order + 1) && no_cycle;
        }

        return no_cycle;
    }

    pub fn reset(&mut self) {
        for i in 0..self.n_nodes {
            self.from_edges[i] = -1;
            self.visit_order[i] = -1;
        }

        for i in 0..self.n_edges {
            self.visited_edges[i] = 0;
        }
    }
}