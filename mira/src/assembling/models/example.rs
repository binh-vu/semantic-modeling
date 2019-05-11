use algorithm::data_structure::graph::Graph;
use assembling::models::variable::TripleVar;
use gmtk::tensors::*;
use algorithm::data_structure::graph::*;
use assembling::auto_label::MRRLabel;
use assembling::models::variable::TripleVarValue;


/// Input to MRR Model
///
/// Variables index are same like link_features index & align with graph edges index
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MRRExample {
    pub sm_idx: usize, // an number associated with an sm, usually the mapping is stored in annotator
    pub graph: Graph,
    pub does_sm_has_hierachy: bool,
    pub sibling_index: SiblingIndex,
    pub variables: Vec<TripleVar>,
    #[serde(skip)]
    pub observed_edge_features: Vec<DenseTensor>, // observed features for Factor-1
    pub root_triple_id: usize, // one triple will be the roots
    #[serde(skip)]
    pub link_features: Vec<LinkFeatures>,
    #[serde(skip)]
    pub node_features: Vec<NodeFeatures>,
    pub label: Option<MRRLabel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkFeatures {
    pub p_triple: Option<f32>,
    pub p_link_given_so: Option<f32>,
    pub p_link_given_s: Option<f32>,
    pub multi_val_prob: Option<f32>,
    pub stype_order: Option<usize>,
    pub delta_stype_score: Option<f32>,
    pub ratio_stype_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeFeatures {
    /// semantic type score of a data node, optional if this is class node
    pub stype_score: Option<f32>,
    pub node_prob: Option<f32>
}

impl MRRExample {
    pub fn new(sm_idx: usize, g: Graph, does_sm_has_hierachy: bool, label: Option<MRRLabel>) -> MRRExample {
        let mut root_triple_id = g.n_edges;
        for n in g.iter_nodes() {
            if n.n_incoming_edges == 0 {
                root_triple_id = n.outgoing_edges[0];
                break;
            }
        }
        debug_assert!(root_triple_id != g.n_edges, "Graph must have a root node");

        MRRExample {
            sm_idx,
            does_sm_has_hierachy,
            sibling_index: SiblingIndex::new(&g),
            variables: Vec::with_capacity(g.n_edges),
            observed_edge_features: Vec::with_capacity(g.n_edges),
            link_features: vec![LinkFeatures::new(); g.n_edges],
            node_features: vec![NodeFeatures::new(); g.n_nodes],
            graph: g,
            label,
            root_triple_id
        }
    }

    pub fn add_var(&mut self, id: usize, val: TripleVarValue, edge_id: usize) {
        self.variables.push(TripleVar::new(id, val, edge_id));
        self.observed_edge_features.push(Default::default());
    }

    #[inline]
    pub fn get_target(&self, var: &TripleVar) -> &Node {
        return self.graph.get_target_node(var.edge_id);
    }
             
    #[inline]
    pub fn get_source(&self, var: &TripleVar) -> &Node {
        return self.graph.get_source_node(var.edge_id);
    }

    #[inline]
    pub fn get_edge(&self, var: &TripleVar) -> &Edge {
        return self.graph.get_edge_by_id(var.edge_id);
    }

    pub fn deserialize(&mut self) {
        // TODO: fix it, override deserialize to do it correctly
        // this has to called everytime before we want to use an deserialized examples
        self.observed_edge_features = vec![Default::default(); self.graph.n_edges];
        self.link_features = vec![LinkFeatures::new(); self.graph.n_edges];
        self.node_features = vec![NodeFeatures::new(); self.graph.n_nodes];
    }
}


impl LinkFeatures {
    pub fn new() -> LinkFeatures {
        LinkFeatures {
            p_triple: None,
            p_link_given_so: None,
            multi_val_prob: None,
            stype_order: None,
            delta_stype_score: None,
            ratio_stype_score: None,
            p_link_given_s: None
        }
    }

    pub fn update(&mut self, p_triple: Option<f32>, p_link_given_so: Option<f32>) {
        self.p_triple = p_triple;
        self.p_link_given_so = p_link_given_so;
    }
}

impl NodeFeatures {
    pub fn new() -> NodeFeatures {
        NodeFeatures {
            stype_score: None,
            node_prob: None
        }
    }
}