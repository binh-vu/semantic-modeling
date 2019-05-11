use fnv::FnvHashSet;
use data_structure::graph::*;

pub struct GraphChildrenIndex {
    value: Vec<FnvHashSet<usize>>
}

impl GraphChildrenIndex {
    pub fn new<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>) -> GraphChildrenIndex {
        let mut value = g.iter_nodes()
            .map(|n| {
                n.iter_outgoing_edges(g)
                    .map(|e| e.target_id)
                    .collect::<FnvHashSet<usize>>()
            })
            .collect::<Vec<_>>();

        GraphChildrenIndex { value }
    }

    pub fn has_child(&self, node_id: usize, child_id: usize) -> bool {
        self.value[node_id].contains(&child_id)
    }
}

