use fnv::FnvHashSet;
use data_structure::graph::*;

/// Stable ancestor index
pub struct AncestorIndex {
    value: Vec<FnvHashSet<usize>>,
    value_iter: Vec<Vec<usize>>,
}

impl AncestorIndex {
    pub fn new<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>) -> AncestorIndex {
        let mut value = vec![FnvHashSet::default(); g.n_nodes];
        let mut value_iter = vec![Vec::new(); g.n_nodes];

        for n in g.iter_class_nodes() {
            let mut stack = n.iter_incoming_edges(g)
                .map(|e| e.get_source_node(g))
                .collect::<Vec<_>>();

            while stack.len() > 0 {
                let cn = stack.pop().unwrap();
                if !value[n.id].contains(&cn.id) {
                    value[n.id].insert(cn.id);
                    value_iter[n.id].push(cn.id);
                }

                for e in cn.iter_incoming_edges(g) {
                    stack.push(e.get_source_node(g));
                }
            }
        }

        AncestorIndex { value, value_iter }
    }

    #[inline]
    pub fn has_ancestor(&self, node_id: usize, ancestor_id: usize) -> bool {
        self.value[node_id].contains(&ancestor_id)
    }

    #[inline]
    pub fn iter_ancestors(&self, node_id: usize) -> &[usize] {
        &self.value_iter[node_id]
    }

    #[inline]
    pub fn get_ancestors(&self, node_id: usize) -> &FnvHashSet<usize> {
        &self.value[node_id]
    }
}


#[cfg(test)]
mod tests {
    use data_structure::graph::graph_util::quick_graph;
    use data_structure::graph::index::AncestorIndex;

    pub fn test_ancestor_label_index() {
//        let g = quick_graph(&[]).0;
//        let index = AncestorIndex::new(&g);

//        assert_eq!(index, vec![
//        ])
    }
}