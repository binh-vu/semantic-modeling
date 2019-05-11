use fnv::FnvHashSet;
use data_structure::graph::*;

pub struct GraphOutgoingEdgeIndex {
    value: Vec<FnvHashSet<usize>>,
}

impl GraphOutgoingEdgeIndex {
    pub fn new<'a, ND: NodeData, ED: EdgeData>(g: &'a Graph<ND, ED>) -> GraphOutgoingEdgeIndex {
        let mut index = g.iter_nodes()
            .map(|n| {
                n.iter_outgoing_edges(g)
                    .map(|e| e.id)
                    .collect::<FnvHashSet<usize>>()
            })
            .collect::<Vec<_>>();

        GraphOutgoingEdgeIndex { value: index }
    }

    #[inline]
    pub fn has_outgoing_edge(&self, node_id: usize, edge_id: usize) -> bool {
        return self.value[node_id].contains(&edge_id)
    }
}