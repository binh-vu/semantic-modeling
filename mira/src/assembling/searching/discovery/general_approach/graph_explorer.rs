use std::ops::Deref;
use std::ops::DerefMut;
use algorithm::data_structure::graph::*;
use std::collections::HashMap;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphExplorer {
    pub graph: Graph,
    pub(super) node_hops: Vec<i32>,
    pub(super) lt0hop_node_ids: Vec<usize>,
    pub(super) eq0hop_node_ids: Vec<usize>,
    pub(super) gt0hop_node_ids: Vec<usize>,
    pub(super) lt0hop_index_node_lbls: HashMap<String, Vec<usize>>,
    pub(super) gt0hop_index_node_lbls: HashMap<String, Vec<usize>>,
}

impl GraphExplorer {
    pub fn new(g: &Graph) -> GraphExplorer {
        GraphExplorer {
            graph: g.clone(),
            node_hops: vec![0; g.n_nodes],
            lt0hop_node_ids: Vec::new(),
            eq0hop_node_ids: (0..g.n_nodes).collect(),
            gt0hop_node_ids: Vec::new(),
            lt0hop_index_node_lbls: Default::default(),
            gt0hop_index_node_lbls: Default::default(),
        }
    }

    pub fn take(g: Graph) -> GraphExplorer {
        GraphExplorer {
            node_hops: vec![0; g.n_nodes],
            eq0hop_node_ids: (0..g.n_nodes).collect(),
            graph: g,
            lt0hop_node_ids: Vec::new(),
            gt0hop_node_ids: Vec::new(),
            lt0hop_index_node_lbls: Default::default(),
            gt0hop_index_node_lbls: Default::default(),
        }
    }

    #[inline]
    pub fn get_hop(&self, node: &Node) -> i32 {
        self.node_hops[node.id]
    }

    #[inline]
    pub fn get_hop_by_id(&self, nid: usize) -> i32 {
        self.node_hops[nid]
    }

    pub fn add_node(&mut self, node: Node, n_hop: i32) -> usize {
        let nid = self.graph.add_node(node);
        self.node_hops.push(n_hop);

        if n_hop == 0 {
            self.eq0hop_node_ids.push(nid);
        } else if n_hop < 0 {
            let n_lbl = &self.graph.get_node_by_id(nid).label;
            self.lt0hop_node_ids.push(nid);
            if !self.lt0hop_index_node_lbls.contains_key(n_lbl) {
                self.lt0hop_index_node_lbls.insert(n_lbl.clone(), Vec::new());
            }
            self.lt0hop_index_node_lbls.get_mut(n_lbl).unwrap().push(nid);
        } else {
            let n_lbl = &self.graph.get_node_by_id(nid).label;
            self.gt0hop_node_ids.push(nid);
            if !self.gt0hop_index_node_lbls.contains_key(n_lbl) {
                self.gt0hop_index_node_lbls.insert(n_lbl.clone(), Vec::new());
            }
            self.gt0hop_index_node_lbls.get_mut(n_lbl).unwrap().push(nid);
        }

        nid
    }

    pub fn iter_eq0hop_nodes(&self) -> IterNode {
        IterNode::new(&self.eq0hop_node_ids, &self.graph)
    }

    pub fn iter_lt0hop_nodes(&self) -> IterNode {
        IterNode::new(&self.lt0hop_node_ids, &self.graph)
    }

    pub fn iter_gt0hop_nodes(&self) -> IterNode {
        IterNode::new(&self.gt0hop_node_ids, &self.graph)
    }

    pub fn iter_lt0hop_nodes_by_label(&self, lbl: &str) -> IterNode {
        if self.lt0hop_index_node_lbls.contains_key(lbl) {
            IterNode::new(&self.lt0hop_index_node_lbls[lbl], &self.graph)
        } else {
            // no matter of what array you are returnning, it won't iterate through it, just need
            // a valid reference
            IterNode::empty(&self.lt0hop_node_ids, &self.graph)
        }
    }

    pub fn iter_gt0hop_nodes_by_label(&self, lbl: &str) -> IterNode {
        if self.gt0hop_index_node_lbls.contains_key(lbl) {
            IterNode::new(&self.gt0hop_index_node_lbls[lbl], &self.graph)
        } else {
            // no matter of what array you are returnning, it won't iterate through it, just need
            // a valid reference
            IterNode::empty(&self.gt0hop_node_ids, &self.graph)
        }
    }
}

impl Deref for GraphExplorer {
    type Target = Graph;

    fn deref(&self) -> &<Self as Deref>::Target {
        &self.graph
    }
}

impl DerefMut for GraphExplorer {
    fn deref_mut(&mut self) -> &mut <Self as Deref>::Target {
        &mut self.graph
    }
}