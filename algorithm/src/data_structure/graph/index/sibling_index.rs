use data_structure::graph::graph::{Graph, NodeData, EdgeData};
use data_structure::graph::Node;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SiblingIndex {
    pub siblings: Vec<Vec<usize>>
}

impl SiblingIndex {
    pub fn new<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>) -> SiblingIndex {
        let mut siblings = vec![Vec::new(); g.n_nodes];
        for n in g.iter_nodes() {
            for e in n.iter_outgoing_edges(g) {
                siblings[e.target_id].reserve_exact(n.n_outgoing_edges - 1);
            }
        }

        for n in g.iter_nodes() {
            for e in n.iter_outgoing_edges(g) {
                for e2 in n.iter_outgoing_edges(g) {
                    if e2.target_id != e.target_id {
                        siblings[e.target_id].push(e2.target_id);
                    }
                }
            }
        }

        SiblingIndex { siblings }
    }

    #[inline]
    pub fn get_n_siblings<ND: NodeData>(&self, n: &Node<ND>) -> usize {
        return self.siblings[n.id].len()
    }
}