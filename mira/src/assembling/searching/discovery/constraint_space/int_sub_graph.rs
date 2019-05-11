use algorithm::data_structure::graph::*;
use assembling::searching::banks::data_structure::int_graph::*;
use super::*;

pub struct IntSubGraph<'a> {
    pub graph: &'a Graph,
    pub int_graph: &'a IntGraph,
    pub bijection: &'a Bijection,
    pub outgoing_edge_index: GraphOutgoingEdgeIndex,
    pub children_index: GraphChildrenIndex,
}

impl<'a> IntSubGraph<'a> {
    pub fn new(graph: &'a Graph, bijection: &'a Bijection, int_graph: &'a IntGraph) -> IntSubGraph<'a> {
        IntSubGraph {
            graph,
            int_graph,
            bijection,
            outgoing_edge_index: GraphOutgoingEdgeIndex::new(graph),
            children_index: GraphChildrenIndex::new(graph),
        }
    }

    /// Iterate through node in int_graph
    pub fn iter_mapped_nodes(&'a self) -> IterISGNode<'a> {
        IterISGNode {
            current_idx: 0,
            sub_graph: self
        }
    }
}

pub struct IterISGNode<'a> {
    current_idx: usize,
    pub(super) sub_graph: &'a IntSubGraph<'a>
}

impl<'a> Iterator for IterISGNode<'a> {
    type Item = &'a IntNode;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.current_idx < self.sub_graph.graph.n_nodes {
            Some(self.sub_graph.int_graph.graph.get_node_by_id(self.sub_graph.bijection.to_x(self.current_idx)))
        } else {
            None
        };

        self.current_idx += 1;
        result
    }
}