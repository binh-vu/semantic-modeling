use super::super::graph::{Graph, NodeData, EdgeData};
use super::super::node::Node;
use data_structure::graph::algorithm::traversal::dfs_full;

/// Detect if graph has cycle.
pub fn has_cycle<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>) -> bool {
    let mut recursion_node = vec![false; g.n_nodes];
    // warning, dfs_full should only run in one-threaded, otherwise, we will hit race-condition
    let recursion_node_ptr = (&mut recursion_node) as *mut Vec<bool>;
    let mut begin_visited = |n: &Node<ND>| {
        if recursion_node[n.id] {
            // back-edge
            return false;
        }

        unsafe { (*recursion_node_ptr)[n.id] = true; }
        return true;
    };
    let mut end_visited = |n: &Node<ND>| {
        unsafe { (*recursion_node_ptr)[n.id] = false; }
        return true;
    };

    let mut has_root_node = false;
    for n in g.iter_nodes() {
        if n.n_incoming_edges == 0 {
            has_root_node = true;
            if !dfs_full(n, g, &mut begin_visited, &mut end_visited) {
                return true;
            }
        }
    }

    return !has_root_node;
}

#[cfg(test)]
mod tests {
    use super::*;
    use data_structure::graph::graph_util::quick_graph;

    #[test]
    pub fn test_has_cycle() {
        let g = quick_graph(&["A----B", "B----C", "A----E", "E----D", "B----D", "D----A"]).0;
        assert!(has_cycle(&g));

        let g = quick_graph(&["A----B", "B----C", "A----E", "E----D", "B----D"]).0;
        assert!(!has_cycle(&g));
    }
}