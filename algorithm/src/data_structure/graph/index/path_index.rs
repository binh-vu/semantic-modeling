use data_structure::graph::*;
use data_structure::matrix::Matrix;

pub struct GraphPathIndex {
    index: Matrix<Vec<Vec<usize>>>
}

impl GraphPathIndex {
    pub fn new<V: NodeData, E: EdgeData>(graph: &Graph<V, E>) -> Option<GraphPathIndex> {
        let mut index: Matrix<Vec<Vec<usize>>> = Matrix::new(graph.n_nodes, graph.n_nodes);
        if algorithm::has_cycle(graph) {
            return None;
        }

        for node in graph.iter_nodes() {
            build_path_by_dfs(node, graph, &mut index);
        }

        Some(GraphPathIndex { index })
    }

    #[inline]
    pub fn get_path_between(&self, source_id: usize, target_id: usize) -> &Vec<Vec<usize>> {
        &self.index[(source_id, target_id)]
    }
}

struct DFS<'a, V: 'a + NodeData> {
    node: &'a Node<V>,
    // endpoint between path of two nodes
    endpoint: (usize, usize),
    // index of paths
    path_idx: usize
}


impl<'a, V: 'a + NodeData> DFS<'a, V> {
    fn get_path(&self, path_index: &'a Matrix<Vec<Vec<usize>>>) -> &Vec<usize> {
        &path_index[self.endpoint][self.path_idx]
    }
}

/// This function build Path Index for graph that doesn't have cycle.
fn build_path_by_dfs<'a, V: 'a + NodeData, E: EdgeData>(source: &'a Node<V>, graph: &'a Graph<V, E>, path_index: &'a mut Matrix<Vec<Vec<usize>>>) {
    let mut stack = Vec::with_capacity(source.n_outgoing_edges);

    for e in source.iter_outgoing_edges(graph) {
        let target = e.get_target_node(graph);
        path_index[(source.id, target.id)].push(vec![e.id]);
        if target.n_outgoing_edges > 0 {
            stack.push(DFS { node: target, endpoint: (source.id, target.id), path_idx: path_index.get(source.id, target.id).len() - 1 });
        }
    }

    while stack.len() > 0 {
        let dfs_node = stack.pop().unwrap();

        for e in dfs_node.node.iter_outgoing_edges(graph) {
            let target = e.get_target_node(graph);

            // compute a path from source into target node and add it into index
            let mut path = dfs_node.get_path(path_index).clone();
            path.push(e.id);

            path_index[(source.id, target.id)].push(path);
            if target.n_outgoing_edges > 0 {
                stack.push(DFS { node: target, endpoint: (source.id, target.id), path_idx: path_index[(source.id, target.id)].len() - 1 });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use data_structure::graph::*;
    use ndarray::arr2;
    use data_structure::graph::graph_util::quick_graph;

    #[test]
    pub fn test_graph_path_index() {
        let g = quick_graph(&["A----B", "B----C", "A----E", "E----D", "B----D"]).0;
        let path_index = GraphPathIndex::new(&g).expect("Acyclic graph");
        assert!(path_index.index.is_equal(&[
            vec![vec![], vec![vec![0]], vec![vec![0, 1]], vec![vec![2]], vec![vec![2, 3], vec![0, 4]]],
            vec![vec![], vec![], vec![vec![1]], vec![], vec![vec![4]]],
            vec![vec![], vec![], vec![], vec![], vec![]],
            vec![vec![], vec![], vec![], vec![], vec![vec![3]]],
            vec![vec![], vec![], vec![], vec![], vec![]]
        ]));
    }

    #[test]
    pub fn test_graph_path_index_cycle() {
        let g = quick_graph(&["A----B", "B----C", "A----E", "E----D", "B----D", "D----A"]).0;
        assert!(GraphPathIndex::new(&g).is_none());
    }
}