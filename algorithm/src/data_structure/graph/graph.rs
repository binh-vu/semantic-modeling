use std::slice::Iter;
use std::fmt;
use serde_json;
use serde_json::Value;
use data_structure::graph::node::NodeType;
use data_structure::graph::edge::EdgeType;
use data_structure::graph::node::Node;
use data_structure::graph::edge::Edge;
use serde::Serialize;
use serde::Serializer;
use serde::Deserialize;
use serde::Deserializer;
use data_structure::graph::graph_iter::{IterEdge, IterNode};
use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::slice::IterMut;


pub trait NodeData: Default + PartialEq + Eq + Clone {
    fn to_dict(&self) -> Value;
    fn from_dict(_: &Value) -> Self;
}

pub trait EdgeData: Default + PartialEq + Eq + Clone {
    fn to_dict(&self) -> Value;
    fn from_dict(_: &Value) -> Self;
}

#[derive(PartialEq, Eq, Default, Clone, Debug)]
pub struct EmptyData {}
impl EdgeData for EmptyData {
    fn to_dict(&self) -> Value {
        Value::Null
    }

    fn from_dict(_: &Value) -> Self {
        EmptyData {}
    }
}
impl NodeData for EmptyData {
    fn to_dict(&self) -> Value {
        Value::Null
    }

    fn from_dict(_: &Value) -> Self {
        EmptyData {}
    }
}


#[derive(Debug, Clone)]
pub struct Graph<ND: NodeData=EmptyData, ED: EdgeData=EmptyData> {
    pub id: String,
    pub n_nodes: usize,
    pub n_edges: usize,
    pub(super) nodes: Vec<Node<ND>>,
    pub(super) edges: Vec<Edge<ED>>,
    pub index_node_type: bool,
    pub index_node_label: bool,
    pub index_edge_label: bool,
    node_index_type: Vec<Vec<usize>>,
    node_index_label: HashMap<String, Vec<usize>>,
    edge_index_label: HashMap<String, Vec<usize>>
}

impl<ND: NodeData, ED: EdgeData> Graph<ND, ED> {

    pub fn new(id: String, index_node_type: bool, index_node_label: bool, index_edge_label: bool) -> Graph<ND, ED> {
        Graph {
            id,
            n_nodes: 0,
            n_edges: 0,
            nodes: Vec::with_capacity(32),
            edges: Vec::with_capacity(31),
            index_node_type,
            index_node_label,
            index_edge_label,
            node_index_type: vec![Vec::new(), Vec::new()],
            node_index_label: Default::default(),
            edge_index_label: Default::default()
        }
    }

    pub fn with_capacity(id: String, estimated_n_nodes: usize, estimated_n_edges: usize, index_node_type: bool, index_node_label: bool, index_edge_label: bool) -> Graph<ND, ED> {
        Graph {
            id,
            n_nodes: 0,
            n_edges: 0,
            nodes: Vec::with_capacity(estimated_n_nodes),
            edges: Vec::with_capacity(estimated_n_edges),
            index_node_type,
            index_node_label,
            index_edge_label,
            node_index_type: vec![Vec::new(), Vec::new()],
            node_index_label: HashMap::with_capacity(estimated_n_nodes),
            edge_index_label: HashMap::with_capacity(estimated_n_edges)
        }
    }

    pub fn new_like(g: &Graph<ND, ED>) -> Graph<ND, ED> {
        Graph::with_capacity(g.id.clone(), g.n_nodes, g.n_edges, g.index_node_type, g.index_node_label, g.index_edge_label)
    }

    /// Merge all nodes & edges of another graph into self
    pub fn update(&mut self, graph: &Graph<ND, ED>) {
        let offset = self.n_nodes;
        for n in graph.iter_nodes() {
            self.add_node(Node::new(n.kind, n.label.clone()));
        }
        for e in graph.iter_edges() {
            self.add_edge(Edge::new(e.kind, e.label.clone(), e.source_id + offset, e.target_id + offset));
        }
    }

    /// Add new node to graph and return its id
    pub fn add_node(&mut self, mut node: Node<ND>) -> usize {
        node.id = self.n_nodes;
        self.n_nodes += 1;

        // update node index
        if self.index_node_label {
            if !self.node_index_label.contains_key(&node.label) {
                self.node_index_label.insert(node.label.clone(), vec![node.id]);
            } else {
                self.node_index_label.get_mut(&node.label).unwrap().push(node.id);
            }
        }
        if self.index_node_type {
            self.node_index_type[node.kind as usize].push(node.id);
        }

        self.nodes.push(node);
        self.n_nodes - 1
    }

    /// add new link to graph and return its id
    pub fn add_edge(&mut self, mut edge: Edge<ED>) -> usize {
        edge.id = self.n_edges;
        self.n_edges += 1;
        self.nodes[edge.source_id].add_outgoing_edge(&edge);
        self.nodes[edge.target_id].add_incoming_edge(&edge);

        // update link index
        if self.index_edge_label {
            if !self.edge_index_label.contains_key(&edge.label) {
                self.edge_index_label.insert(edge.label.clone(), vec![edge.id]);
            } else {
                self.edge_index_label.get_mut(&edge.label).unwrap().push(edge.id);
            }

        }

        self.edges.push(edge);
        self.n_edges - 1
    }

    /// Remove the edge has biggest id in the graph
    pub fn remove_last_edge(&mut self) -> Edge<ED> {
        let edge = self.edges.pop().unwrap();
        self.n_edges -= 1;
        self.nodes[edge.source_id].remove_last_outgoing_edge();
        self.nodes[edge.target_id].remove_last_incoming_edge();

        if self.index_edge_label {
            self.edge_index_label.get_mut(&edge.label).unwrap().pop();
        }

        edge
    }

    /// Remove the node has biggest id in the graph
    pub fn remove_last_node(&mut self) -> Node<ND> {
        let node = self.nodes.pop().unwrap();
        assert_eq!(node.n_incoming_edges + node.n_outgoing_edges, 0, "Cannot remove node has edges");

        self.n_nodes -= 1;

        if self.index_node_label {
            self.node_index_label.get_mut(&node.label).unwrap().pop();
        }

        if self.index_node_type {
            self.node_index_type[node.kind as usize].pop();
        }

        node
    }

    pub fn get_n_class_nodes(&self) -> usize {
        debug_assert!(self.index_node_type);
        self.node_index_type[NodeType::ClassNode as usize].len()
    }

    pub fn get_n_data_nodes(&self) -> usize {
        debug_assert!(self.index_node_type);
        self.node_index_type[NodeType::DataNode as usize].len()
    }

    #[inline]
    pub fn get_node_by_id(&self, idx: usize) -> &Node<ND> { &self.nodes[idx] }

    #[inline]
    pub fn get_edge_by_id(&self, idx: usize) -> &Edge<ED> { &self.edges[idx] }

    #[inline]
    pub fn get_mut_node_by_id(&mut self, idx: usize) -> &mut Node<ND> { &mut self.nodes[idx] }

    #[inline]
    pub fn get_mut_edge_by_id(&mut self, idx: usize) -> &mut Edge<ED> { &mut self.edges[idx] }

    #[inline]
    pub fn has_node_with_id(&self, id: usize) -> bool { id < self.n_nodes }

    #[inline]
    pub fn has_edge_with_id(&self, id: usize) -> bool { id < self.n_edges }

    #[inline]
    pub fn iter_nodes(&self) -> Iter<Node<ND>> { self.nodes.iter() }

    #[inline]
    pub fn iter_edges(&self) -> Iter<Edge<ED>> { return self.edges.iter() }

    #[inline]
    pub fn iter_mut_edges(&mut self) -> IterMut<Edge<ED>> { return self.edges.iter_mut() }

    pub fn get_first_root_node(&self) -> Option<&Node<ND>> {
        for n in &self.nodes {
            if n.n_incoming_edges == 0 {
                return Some(n);
            }
        }

        return None;
    }

    pub fn get_first_node_by_label(&self, lbl: &str) -> &Node<ND> {
        &self.nodes[self.node_index_label[lbl][0]]
    }

    pub fn iter_class_nodes(&self) -> IterNode<ND, ED> {
        debug_assert!(self.index_node_type);
        IterNode::new(&self.node_index_type[NodeType::ClassNode as usize], self)
    }

    pub fn iter_data_nodes(&self) -> IterNode<ND, ED> {
        debug_assert!(self.index_node_type);
        IterNode::new(&self.node_index_type[NodeType::DataNode as usize], self)
    }

    pub fn iter_nodes_by_label(&self, lbl: &str) -> IterNode<ND, ED> {
        debug_assert!(self.index_node_label);
        if self.node_index_label.contains_key(lbl) {
            IterNode::new(&self.node_index_label[lbl], self)
        } else {
            IterNode::empty(&self.node_index_type[0], self)
        }
    }

    pub fn iter_edges_by_label(&self, lbl: &str) -> IterEdge<ND, ED> {
        debug_assert!(self.index_edge_label);
        if self.edge_index_label.contains_key(lbl) {
            IterEdge::new(&self.edge_index_label[lbl], self)
        } else {
            IterEdge::empty(&self.node_index_type[0], self)
        }
    }

    #[inline]
    pub fn get_target_node(&self, eid: usize) -> &Node<ND> {
        &self.nodes[self.edges[eid].target_id]
    }

    #[inline]
    pub fn get_source_node(&self, eid: usize) -> &Node<ND> {
        &self.nodes[self.edges[eid].source_id]
    }

    pub fn to_dict(&self) -> Value {
        let obj: Value = json!({
            "name": self.id,
            "index_node_type": self.index_node_type,
            "index_node_label": self.index_node_label,
            "index_link_label": self.index_edge_label,
            "nodes": Value::Array(self.nodes.iter().map(|n| n.to_dict()).collect()),
            "links": Value::Array(self.edges.iter().map(|e| e.to_dict()).collect()),
        });

        return obj;
    }

    /// Load graph from json
    pub fn from_dict(obj: &Value) -> Graph<ND, ED> {
        let mut graph = Graph::with_capacity(
            obj["name"].as_str().unwrap().to_owned(),
            obj["nodes"].as_array().unwrap().len(),
            obj["links"].as_array().unwrap().len(),
            obj["index_node_type"].as_bool().unwrap_or(false),
            obj["index_node_label"].as_bool().unwrap_or(false),
            obj["index_link_label"].as_bool().unwrap_or(false)
        );

        for node in obj["nodes"].as_array().unwrap().iter() {
            let nid = graph.add_node(Node::new_with_data(
                NodeType::new(node["type"].as_i64().unwrap() as i8),
                node["label"].as_str().unwrap().to_owned(),
                ND::from_dict(&node["data"])
            ));
            assert_eq!(node["id"].as_i64().unwrap() as usize, nid);
        }

        for link in obj["links"].as_array().unwrap().iter() {
            let eid = graph.add_edge(Edge::new_with_data(
                EdgeType::new(link["type"].as_i64().unwrap() as i8),
                link["label"].as_str().unwrap().to_owned(),
                link["source_id"].as_i64().unwrap() as usize,
                link["target_id"].as_i64().unwrap() as usize,
                ED::from_dict(&link["data"])
            ));
            assert_eq!(link["id"].as_i64().unwrap() as usize, eid);
        }

        return graph;
    }

}

impl<ND: NodeData, ED: EdgeData> fmt::Display for Graph<ND, ED> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Graph(id={}\n", self.id)?;
        for e in self.iter_edges() {
            write!(f, "\t+ {}---{}---{}\n", self.get_node_by_id(e.source_id).label, e.label, self.get_node_by_id(e.target_id).label)?;
        }

        write!(f, ")")
    }
}

impl<ND: NodeData, ED: EdgeData> Serialize for Graph<ND, ED> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where S: Serializer {

        let value: Value = self.to_dict();
        value.serialize(serializer)
    }
}

impl<'de, ND: NodeData, ED: EdgeData> Deserialize<'de> for Graph<ND, ED> {
    fn deserialize<D>(deserializer: D) -> Result<Graph<ND, ED>, D::Error>
        where D: Deserializer<'de> {

        let result = Value::deserialize(deserializer);
        match result {
            Ok(val) => Ok(Graph::from_dict(&val)),
            Err(e) => Err(e)
        }
    }
}

impl<ND: NodeData, ED: EdgeData> PartialEq for Graph<ND, ED> {
    fn eq(&self, other: &Graph<ND, ED>) -> bool {
        if self.n_nodes != other.n_nodes || self.n_edges != other.n_edges {
            return false;
        }

        for n in self.iter_nodes() {
            if n != other.get_node_by_id(n.id) {
                return false;
            }
        }

        for e in self.iter_edges() {
            if e != other.get_edge_by_id(e.id) {
                return false;
            }
        }

        return true;
    }
}

impl<ND: NodeData, ED: EdgeData> Eq for Graph<ND, ED> {}