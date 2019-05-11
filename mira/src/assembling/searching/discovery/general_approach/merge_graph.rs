use algorithm::data_structure::graph::*;
use super::merge_plan::*;

pub struct MergeGraph<'a> {
    g_part_a: &'a Graph,
    g_part_b: &'a Graph,
    pub g_part_merge: Graph,

    pub point_a: IntegrationPoint,
    pub point_b: IntegrationPoint
}

impl<'a> MergeGraph<'a> {
    pub fn new(g_part_a: &'a Graph, g_part_b: &'a Graph, g_part_merge: Graph, point_a: IntegrationPoint, point_b: IntegrationPoint) -> MergeGraph<'a> {
        MergeGraph {
            g_part_a,
            g_part_b,
            g_part_merge,
            point_a,
            point_b
        }
    }

    #[inline]
    pub fn get_n_nodes(&self) -> usize {
        self.g_part_a.n_nodes + self.g_part_b.n_nodes + self.g_part_merge.n_nodes - 2
    }

    #[inline]
    pub fn get_n_edges(&self) -> usize {
        self.g_part_a.n_edges + self.g_part_b.n_edges + self.g_part_merge.n_edges
    }

    pub fn proceed_merging(&self) -> Graph {
        let mut g = Graph::with_capacity(
            "".to_owned(), 
            self.g_part_a.n_nodes + self.g_part_b.n_nodes + self.g_part_merge.n_nodes - 2,
            self.g_part_a.n_edges + self.g_part_b.n_edges + self.g_part_merge.n_edges,
            self.g_part_a.index_node_type || self.g_part_b.index_node_type,
            self.g_part_a.index_node_label || self.g_part_b.index_node_label,
            self.g_part_a.index_edge_label || self.g_part_b.index_edge_label
        );

        g.update(self.g_part_a);
        g.update(self.g_part_b);

        let n_offset = g.n_nodes;

        let mut source_id;
        let mut target_id;

        // notice that there are two nodes in part_merge got removed,
        // those two ndoes are always stored at the end of part_merge node's array
        // so we just need to skip them
        for i in 0..(self.g_part_merge.n_nodes - 2) {
            let n = self.g_part_merge.get_node_by_id(i);
            g.add_node(Node::new(n.kind, n.label.clone()));
        }

        for e in self.g_part_merge.iter_edges() {
            if e.id == self.point_a.edge_id || e.id == self.point_b.edge_id {
                if e.id == self.point_a.edge_id {
                    if self.point_a.is_incoming_edge {
                        // compute source_id because part_a doesn't have source node
                        source_id = if self.g_part_merge.n_edges > 1 {
                            // because part_merge is actually a linear chain, so when number of links > 1, 
                            // self.point_a.source is not an integration point
                            self.point_a.source_id + n_offset
                        } else {
                            // self.point_a.source is an integration point, so source_id is self.point_b.source_id
                            // moved by some units
                            self.point_b.source_id + self.g_part_a.n_nodes
                        };
                        target_id = self.point_a.target_id
                    } else {
                        // compute target_id because part_a doesn't have target node
                        target_id = if self.g_part_merge.n_edges > 1 {
                            // because part_merge is actually a linear-chain, so when number of links > 1
                            // self.point_a.target is not an integration point
                            self.point_a.target_id + n_offset
                        } else {
                            // self.point_a.target is an integration point, so target_id is self.point_b.target_id 
                            // moved by some units
                            self.point_b.target_id + self.g_part_a.n_nodes
                        };
                        source_id = self.point_a.source_id;
                    }
                } else {
                    if self.point_b.is_incoming_edge {
                        // compute source_id because part_b doesn't have source node, same like self.point_a
                        source_id = if self.g_part_merge.n_edges > 1 {
                            self.point_b.source_id + n_offset
                        } else {
                            self.point_a.source_id
                        };
                        target_id = self.point_b.target_id + self.g_part_a.n_nodes;
                    } else {
                        // compute target_id because part_b doesn't have target node, same like self.point_a
                        target_id = if self.g_part_merge.n_edges > 1 {
                            self.point_b.target_id + n_offset
                        } else {
                            self.point_a.target_id
                        };
                        source_id = self.point_b.source_id + self.g_part_a.n_nodes;
                    }
                }

                g.add_edge(Edge::new(e.kind, e.label.clone(), source_id, target_id));
            } else {
                g.add_edge(Edge::new(
                    e.kind, e.label.clone(),
                    e.source_id + n_offset, e.target_id + n_offset
                ));
            }
        }

        g
    }
}