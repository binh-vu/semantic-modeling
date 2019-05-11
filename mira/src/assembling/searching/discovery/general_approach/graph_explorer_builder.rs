use std::collections::HashMap;
use algorithm::data_structure::graph::*;
use fnv::FnvHashMap;
use assembling::searching::discovery::general_approach::triple_adviser::TripleAdviser;
use assembling::searching::discovery::general_approach::graph_explorer::GraphExplorer;

// Use ot build graph explorer for one data source
pub struct GraphExplorerBuilder<'a> {
    max_data_node_hop: usize,
    max_class_node_hop: usize,
    triple_adviser: &'a mut TripleAdviser,
    explored_data_nodes: HashMap<String, GraphExplorer>,
    explored_class_nodes: HashMap<String, GraphExplorer>
}

impl<'a> GraphExplorerBuilder<'a> {
    pub fn new(triple_adviser: &'a mut TripleAdviser, max_data_node_hop: usize, max_class_node_hop: usize) -> GraphExplorerBuilder<'a> {
        GraphExplorerBuilder {
            max_data_node_hop,
            max_class_node_hop,
            triple_adviser,
            explored_data_nodes: Default::default(),
            explored_class_nodes: Default::default(),
        }
    }

    pub fn build(&mut self, g: &Graph) -> GraphExplorer {
        let max_class_node_hop = self.max_class_node_hop as i32;
        let max_data_node_hop = self.max_data_node_hop as i32;

        let mut g_explorer = GraphExplorer::new(g);

        for n in g.iter_nodes() {
            // if current_node_label is not cached, init and then cache it
            let sub_g_explorer = if n.is_class_node() {
                if !self.explored_class_nodes.contains_key(&n.label) {
                    let _temp = self.explore_node(n, max_class_node_hop);
                    self.explored_class_nodes.insert(n.label.clone(), _temp);
                }

                &self.explored_class_nodes[&n.label]
            } else {
                if !self.explored_data_nodes.contains_key(&n.label) {
                    let _temp = self.explore_node(n, max_data_node_hop);
                    self.explored_data_nodes.insert(n.label.clone(), _temp);
                }

                &self.explored_data_nodes[&n.label]
            };

            // now pour content of explored_graph to g_explorer
            if n.n_incoming_edges == 0 {
                // to maintain tree structure, it can only expand if it doesn't have any parent
                // explore incoming nodes
                let mut current_hops = vec![0]; // node 0 always in central hop
                let mut id_map: FnvHashMap<usize, usize> = Default::default();
                id_map.insert(0, n.id);
                let mut next_hops = Vec::new();

                while current_hops.len() > 0 {
                    for curr_node_id in current_hops {
                        for incoming_edge in sub_g_explorer.get_node_by_id(curr_node_id).iter_incoming_edges(sub_g_explorer) {
                            let source_node = sub_g_explorer.get_node_by_id(incoming_edge.source_id);
                            id_map.insert(source_node.id, g_explorer.add_node(Node::new(source_node.kind, source_node.label.clone()), sub_g_explorer.get_hop(source_node)));
                            g_explorer.add_edge(Edge::new(
                                incoming_edge.kind.clone(),
                                incoming_edge.label.clone(),
                                id_map[&incoming_edge.source_id],
                                id_map[&incoming_edge.target_id],
                            ));

                            next_hops.push(source_node.id);
                        }
                    }

                    current_hops = next_hops;
                    next_hops = Vec::new();
                }
            }

            // explore outgoing nodes
            {
                let mut current_hops = vec![0]; // node 0 always in central hop
                let mut id_map: FnvHashMap<usize, usize> = Default::default();
                id_map.insert(0, n.id);
                let mut next_hops = Vec::new();

                while current_hops.len() > 0 {
                    for curr_node_id in current_hops {
                        for outgoing_edge in sub_g_explorer.get_node_by_id(curr_node_id).iter_outgoing_edges(sub_g_explorer) {
                            let target_node = sub_g_explorer.get_node_by_id(outgoing_edge.target_id);
                            id_map.insert(target_node.id, g_explorer.add_node(Node::new(target_node.kind, target_node.label.clone()), sub_g_explorer.get_hop(target_node)));
                            g_explorer.add_edge(Edge::new(
                                outgoing_edge.kind.clone(),
                                outgoing_edge.label.clone(),
                                id_map[&outgoing_edge.source_id],
                                id_map[&outgoing_edge.target_id],
                            ));

                            next_hops.push(target_node.id);
                        }
                    }

                    current_hops = next_hops;
                    next_hops = Vec::new();
                }
            }
        }

        g_explorer
    }

    fn explore_node(&mut self, n: &Node, max_hop: i32) -> GraphExplorer {
        let mut _g = Graph::new("".to_owned(), false, false, false);
        _g.add_node(Node::new(n.kind, n.label.clone()));
        let mut g = GraphExplorer::take(_g);

        let mut current_hops = vec![0];
        let mut next_hops = Vec::new();

        // there are many possible links between two nodes, we want only one link between 2 nodes, so
        // we are going to have so many same labeled nodes in explore graph
        // add outgoing nodes, positive hop
        for hop in 0..max_hop {
            for curr_node_id in current_hops {
                for (e_lbl, target_lbl) in self.triple_adviser.get_pred_objs(&g.get_node_by_id(curr_node_id).label) {
                    let next_node_id = g.add_node(Node::new(NodeType::ClassNode, target_lbl.clone()), hop + 1);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e_lbl.clone(), curr_node_id, next_node_id));

                    next_hops.push(next_node_id);
                }
            }

            current_hops = next_hops;
            next_hops = Vec::new();
        }

        current_hops = vec![0];
        next_hops.clear();

        // add incoming nodes, negative hop
        for hop in 0..max_hop {
            for curr_node_id in current_hops {
                for (source_lbl, e_lbl) in self.triple_adviser.get_subj_preds(&g.get_node_by_id(curr_node_id).label) {
                    let next_node_id = g.add_node(Node::new(NodeType::ClassNode, source_lbl.clone()), -hop - 1);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e_lbl.clone(), next_node_id, curr_node_id));
                    next_hops.push(next_node_id);
                }
            }

            current_hops = next_hops;
            next_hops = Vec::new();
        }

        g
    }
}
