use serde_json;
use algorithm::data_structure::graph::*;
use rdb2rdf::models::semantic_model::SemanticModel;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::BTreeMap;
use fnv::FnvHashMap;
use rdb2rdf::models::semantic_model::Attribute;
use rdb2rdf::ontology::ont_graph::OntGraph;
use serde_json::Value;
use fnv::FnvHashSet;


pub const TAG_FROM_NEW_SOURCE: &str = "TAG_FROM_NEW_SOURCE";
pub const TAG_FROM_ONTOLOGY: &str = "TAG_FROM_ONTOLOGY";


#[derive(Clone, Serialize, Debug, Default, PartialEq, Eq)]
pub struct INodeData {
    pub tags: HashSet<String>,
    // name of this data nodes in previous models
    pub original_names: HashMap<String, String>,
    // id of this node in previouos models
    pub original_ids: HashMap<String, usize>
}

#[derive(Clone, Serialize, Debug, Default)]
pub struct IEdgeData {
    pub tags: HashSet<String>,
    pub weight: f32
}

impl PartialEq for IEdgeData {
    fn eq(&self, other: &IEdgeData) -> bool {
        self.tags == other.tags && self.weight == other.weight
    }
}

impl Eq for IEdgeData {}

impl INodeData {
    pub fn new(tag: String) -> INodeData {
        let mut tags: HashSet<String> = Default::default();
        tags.insert(tag);

        INodeData {
            tags,
            original_names: Default::default(),
            original_ids: Default::default()
        }
    }

    pub fn new_with_provenance(tag: String, id: usize, name: String) -> INodeData {
        let mut tags: HashSet<String> = Default::default();
        tags.insert(tag.clone());
        let mut original_names: HashMap<String, String> = Default::default();
        original_names.insert(tag.clone(), name);
        let mut original_ids: HashMap<String, usize> = Default::default();
        original_ids.insert(tag, id);

        INodeData {
            tags,
            original_names,
            original_ids
        }
    }
}

impl IEdgeData {
    pub fn new(tag: String) -> IEdgeData {
        let mut tags: HashSet<String> = Default::default();
        tags.insert(tag);
        IEdgeData {
            tags,
            weight: 0.0
        }
    }

    pub fn new_with_weight(tag: String, weight: f32) -> IEdgeData {
        let mut tags: HashSet<String> = Default::default();
        tags.insert(tag);
        IEdgeData { tags, weight }
    }

    pub fn overlaps_tag(&self, other: &IEdgeData) -> bool {
        for tag in &other.tags {
            if self.tags.contains(tag) {
                return true;
            }
        }
        return false;
    }
}

impl NodeData for INodeData {
    fn to_dict(&self) -> Value {
        json!({
            "tags": self.tags,
            "original_names": self.original_names,
            "original_ids": self.original_ids
        })
    }

    fn from_dict(val: &Value) -> Self {
        INodeData {
            tags: serde_json::from_value(val["tags"].clone()).unwrap(),
            original_names: serde_json::from_value(val["original_names"].clone()).unwrap(),
            original_ids: serde_json::from_value(val["original_ids"].clone()).unwrap()
        }
    }
}

impl EdgeData for IEdgeData {
    fn to_dict(&self) -> Value {
        json!({
            "tags": self.tags,
            "weight": self.weight
        })
    }

    fn from_dict(val: &Value) -> Self {
        IEdgeData {
            tags: serde_json::from_value(val["tags"].clone()).unwrap(),
            weight: val["weight"].as_f64().unwrap() as f32
        }
    }
}

pub type IntEdge = Edge<IEdgeData>;
pub type IntNode = Node<INodeData>;

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, Debug)]
pub struct IntGraph {
    pub graph: Graph<INodeData, IEdgeData>,
}

impl IntGraph {
    pub fn new(train_sms: &[&SemanticModel]) -> IntGraph {
        // 1st STEP: add_known_models
        let mut int_graph = IntGraph {
            graph: Graph::with_capacity("Integration Graph".to_owned(), 100, 100, true, true, true),
        };

        add_known_models(&mut int_graph, train_sms);
        int_graph
    }

    pub fn adapt_new_source(&self, sm: &SemanticModel, ont_graph: Option<&OntGraph>) -> IntGraph {
        let mut g = self.clone();
        add_semantic_types(&mut g, &sm.attrs);
        if let Some(ont_graph) = ont_graph {
            add_ont_graphs(&mut g, ont_graph);
        }

        return g;
    }

    pub fn earse_weights(&mut self) {
        for e in self.graph.iter_mut_edges() {
            e.data.weight = 0.0;
        }
    }

    pub fn to_normal_graph(&self) -> Graph {
        let mut g = Graph::with_capacity(self.graph.id.clone(), self.graph.n_nodes, self.graph.n_edges, true, true, true);

        for n in self.graph.iter_nodes() {
            g.add_node(Node::new(n.kind.clone(), n.label.clone()));
        }

        for e in self.graph.iter_edges() {
            g.add_edge(Edge::new(e.kind.clone(), e.label.clone(), e.source_id, e.target_id));
        }

        return g;
    }
}

fn add_known_models(int_graph: &mut IntGraph, train_sms: &[&SemanticModel]) {
    let int_graph_ptr = int_graph as *mut IntGraph;

    for sm in train_sms.iter() {
        let mut H: FnvHashMap<usize, usize> = Default::default();
        let mut vertices: BTreeMap<&String, Vec<&Node>> = Default::default();

        for v in sm.graph.iter_class_nodes() {
            if !vertices.contains_key(&v.label) {
                vertices.insert(&v.label, Default::default());
            }
            vertices.get_mut(&v.label).unwrap().push(v);
        }

        for (&lbl, ref lbl_group) in vertices.iter() {
            let c1 = lbl_group.len();
            let c2 = int_graph.graph.iter_nodes_by_label(lbl).count();

            for i in c2..c1 {
                int_graph.graph.add_node(Node::new(NodeType::ClassNode, lbl.to_owned()));
            }

            // step 1, try to match node with same parents first (same parent have higher priority)
            let matched_nodes = int_graph.graph.iter_nodes_by_label(lbl).collect::<Vec<_>>();

            for v in lbl_group.iter() {
                let parent_labels = v.iter_incoming_edges(&sm.graph)
                    .map(|e| &e.get_source_node(&sm.graph).label)
                    .collect::<HashSet<_>>();

                let siblings_unmapped_nodes = matched_nodes.iter()
                    .filter(|&&n| {
                        if !n.data.tags.contains(&sm.id) {
                            return n
                                .iter_incoming_edges(&int_graph.graph)
                                .any(|e| parent_labels.contains(&e.get_source_node(&int_graph.graph).label));
                        } else {
                            return false;
                        }
                    })
                    .cloned().collect::<Vec<_>>();

                if siblings_unmapped_nodes.len() > 0 {
                    let v_prime = siblings_unmapped_nodes
                        .iter()
                        .max_by_key(|&&n| n.data.tags.len())
                        .unwrap();
                    H.insert(v.id, v_prime.id);
                    unsafe {
                        let mut ndata = &mut (&mut *int_graph_ptr).graph.get_mut_node_by_id(v_prime.id).data;
                        ndata.tags.insert(sm.id.clone());
                        ndata.original_ids.insert(sm.id.clone(), v.id);
                    }
                }
            }

            // step 2, now we match the rest
            for v in lbl_group.iter() {
                if H.contains_key(&v.id) {
                    continue;
                }

                let v_prime = matched_nodes.iter()
                    .filter(|&&n| !n.data.tags.contains(&sm.id))
                    .max_by_key(|&&n| n.data.tags.len())
                    .unwrap();

                H.insert(v.id, v_prime.id);
                unsafe {
                    let mut ndata = &mut (&mut *int_graph_ptr).graph.get_mut_node_by_id(v_prime.id).data;
                    ndata.tags.insert(sm.id.clone());
                    ndata.original_ids.insert(sm.id.clone(), v.id);
                }
            }
        }

        // use in case a class node have two data nodes, which have same predicate label
        let mut used_v_prime_id: FnvHashSet<usize> = Default::default();
        for e in sm.graph.iter_edges() {
            let u = e.get_source_node(&sm.graph);
            let v = e.get_target_node(&sm.graph);

            if !(u.is_class_node() && v.is_data_node()) {
                continue;
            }

            let v_prime_id = {
                // they can also have two predicates point to data node
                let u_prime = int_graph.graph.get_node_by_id(H[&u.id]);
                match u_prime.iter_outgoing_edges(&int_graph.graph).find(|e_prime| {
                    e_prime.label == e.label && !used_v_prime_id.contains(&e_prime.target_id) && e_prime.get_target_node(&int_graph.graph).is_data_node()
                }) {
                    None => {
                        unsafe {
                            (&mut *int_graph_ptr).graph.add_node(Node::new(NodeType::DataNode, "DATA_NODE".to_owned()))
                        }
                    },
                    Some(e_prime) => e_prime.target_id
                }
            };

            used_v_prime_id.insert(v_prime_id);
            H.insert(v.id, v_prime_id);
            {
                let mut ndata = &mut int_graph.graph.get_mut_node_by_id(v_prime_id).data;
                ndata.tags.insert(sm.id.clone());
                ndata.original_names.insert(sm.id.clone(), v.label.clone());
                ndata.original_ids.insert(sm.id.clone(), v.id);
            }
        }

        for e in sm.graph.iter_edges() {
            let u = e.get_source_node(&sm.graph);
            let v = e.get_target_node(&sm.graph);

            let e_prime_id = {
                let u_prime = int_graph.graph.get_node_by_id(H[&u.id]);
                let v_prime = int_graph.graph.get_node_by_id(H[&v.id]);

                match v_prime.iter_incoming_edges(&int_graph.graph).find(|e_prime| e_prime.source_id == u_prime.id && e_prime.label == e.label) {
                    None => {
                        unsafe {
                            (&mut *int_graph_ptr).graph.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), u_prime.id, v_prime.id))
                        }
                    },
                    Some(e_prime) => e_prime.id
                }
            };

            int_graph.graph.get_mut_edge_by_id(e_prime_id).data.tags.insert(sm.id.clone());
        }
    }
}

fn add_semantic_types(graph: &mut IntGraph, attrs: &[Attribute]) {
    let graph_ptr = graph as *mut IntGraph;

    for attr in attrs.iter() {
        for st in &attr.semantic_types {
            if graph.graph.iter_nodes_by_label(&st.class_uri).next().is_none() {
                graph.graph.add_node(Node::new_with_data(NodeType::ClassNode, st.class_uri.clone(), INodeData::new("NEW_DATA_SOURCE".to_owned())));
            }

            for v in graph.graph.iter_nodes_by_label(&st.class_uri) {
                if !v.iter_outgoing_edges(&graph.graph).any(|e| e.label == st.predicate && e.get_target_node(&graph.graph).is_data_node()) {
                    unsafe {
                        let wid = (&mut *graph_ptr).graph.add_node(Node::new_with_data(
                            NodeType::DataNode,
                            "DATA_NODE".to_owned(), 
                            INodeData::new_with_provenance(TAG_FROM_NEW_SOURCE.to_owned(), attr.id, attr.label.clone())
                        ));
                        (&mut *graph_ptr).graph.add_edge(Edge::new_with_data(
                            EdgeType::Unspecified,
                            st.predicate.clone(),
                            v.id, wid,
                            IEdgeData::new(TAG_FROM_NEW_SOURCE.to_owned())));
                    }
                }
            }
        }
    }
}

fn add_ont_graphs(graph: &mut IntGraph, ont_graph: &OntGraph) {
    let graph_ptr = graph as *mut IntGraph;

    for u in graph.graph.iter_class_nodes() {
        for v in graph.graph.iter_class_nodes() {
            if u.id == v.id {
                continue;
            }

            for p in ont_graph.get_possible_predicates(&u.label, &v.label) {
                if let None = v.iter_incoming_edges(&graph.graph).find(|e| e.source_id == u.id && e.label == p.uri) {
                    unsafe {
                        (&mut *graph_ptr).graph.add_edge(Edge::new_with_data(EdgeType::Unspecified, p.uri.clone(), u.id, v.id, IEdgeData::new(TAG_FROM_ONTOLOGY.to_owned())));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assembling::tests::tests::*;

    /// Repeat creating IntGraph would produce same result
    #[test]
    pub fn test_build_int_graph_stable() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let int_graph = IntGraph::new(&input.get_train_sms());

        let gold_file = format!("resources/assembling/searching/banks/int_graph.json");
//        serialize_json(&int_graph, &gold_file);

        let gold_int_graph: IntGraph = deserialize_json(&gold_file);
        assert_eq!(int_graph.graph, gold_int_graph.graph);
    }

    #[test]
    pub fn test_add_semantic_types() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let int_graph = IntGraph::new(&input.get_train_sms());

        let mut new_int_graph = int_graph.clone();
        add_semantic_types(&mut new_int_graph, &input.get_train_sms()[0].attrs);

        for n in new_int_graph.graph.iter_nodes() {
            if !int_graph.graph.has_node_with_id(n.id) {
                assert!(n.is_data_node());
            }
        }
    }
}
