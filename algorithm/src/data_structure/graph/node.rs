use data_structure::graph::edge::Edge;
use serde_json::Value;
use data_structure::graph::graph::{ Graph, NodeData, EdgeData, EmptyData };
use data_structure::graph::graph_iter::*;
use data_structure::graph::SiblingIndex;

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum NodeType {
    ClassNode = 0,
    DataNode = 1,
}

impl NodeType {
    pub fn new(val: i8) -> NodeType {
        match val {
            1 => NodeType::ClassNode,
            2 => NodeType::DataNode,
            _ => panic!("Invalid type: {}", val)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<D: NodeData=EmptyData> {
    pub id: usize,
    pub kind: NodeType,
    pub label: String,
    pub n_incoming_edges: usize,
    pub n_outgoing_edges: usize,
    pub incoming_edges: Vec<usize>,
    pub outgoing_edges: Vec<usize>,
    pub data: D
}


impl<D: NodeData> Node<D> {
    pub fn new(kind: NodeType, label: String) -> Node<D> {
        Node {
            id: 12938012983,
            kind,
            label,
            n_incoming_edges: 0,
            n_outgoing_edges: 0,
            incoming_edges: Vec::new(),
            outgoing_edges: Vec::new(),
            data: Default::default()
        }
    }

    pub fn new_with_data(kind: NodeType, label: String, data: D) -> Node<D> {
        Node {
            id: 12938012983,
            kind,
            label,
            n_incoming_edges: 0,
            n_outgoing_edges: 0,
            incoming_edges: Vec::new(),
            outgoing_edges: Vec::new(),
            data
        }
    }

    #[inline]
    pub fn is_data_node(&self) -> bool { self.kind == NodeType::DataNode }

    #[inline]
    pub fn is_class_node(&self) -> bool { self.kind == NodeType::ClassNode }

    pub fn add_incoming_edge<ED: EdgeData>(&mut self, link: &Edge<ED>) {
        self.incoming_edges.push(link.id);
        self.n_incoming_edges += 1;
    }

    pub fn add_outgoing_edge<ED: EdgeData>(&mut self, link: &Edge<ED>) {
        self.outgoing_edges.push(link.id);
        self.n_outgoing_edges += 1;
    }

    pub(super) fn remove_last_outgoing_edge(&mut self) -> usize {
        self.n_outgoing_edges -= 1;
        self.outgoing_edges.pop().unwrap()
    }

    pub(super) fn remove_last_incoming_edge(&mut self) -> usize {
        self.n_incoming_edges -= 1;
        self.incoming_edges.pop().unwrap()
    }

    pub fn first_incoming_edge<'a, ED: EdgeData>(&self, graph: &'a Graph<D, ED>) -> Option<&'a Edge<ED>> {
        if self.n_incoming_edges == 0 {
            None
        } else {
            Some(&graph.edges[self.incoming_edges[0]])
        }
    }

    pub fn first_outgoing_edge<'a, ED: EdgeData>(&self, graph: &'a Graph<D, ED>) -> Option<&'a Edge<ED>> {
        if self.n_outgoing_edges == 0 {
            None
        } else {
            Some(&graph.edges[self.outgoing_edges[0]])
        }
    }

    pub fn first_parent<'a, ED: EdgeData>(&self, graph: &'a Graph<D, ED>) -> Option<&'a Node<D>> {
        if self.n_incoming_edges == 0 {
            None
        } else {
            Some(&graph.nodes[graph.edges[self.incoming_edges[0]].source_id])
        }
    }

    pub fn first_child<'a, ED: EdgeData>(&self, graph: &'a Graph<D, ED>) -> Option<&'a Node<D>> {
        if self.n_outgoing_edges == 0 {
            None
        } else {
            Some(&graph.nodes[graph.edges[self.outgoing_edges[0]].target_id])
        }
    }

    pub fn iter_incoming_edges<'a, ED: EdgeData>(&'a self, graph: &'a Graph<D, ED>) -> IterEdge<'a, D, ED> {
        IterEdge::new(&self.incoming_edges, graph)
    }

    pub fn iter_outgoing_edges<'a, ED: EdgeData>(&'a self, graph: &'a Graph<D, ED>) -> IterEdge<'a, D, ED> {
        IterEdge::new(&self.outgoing_edges, graph)
    }

    pub fn iter_children<'a, ED: EdgeData>(&'a self, graph: &'a Graph<D, ED>) -> IterChildrenNode<'a, D, ED> {
        IterChildrenNode::new(&self.outgoing_edges, graph)
    }

    pub fn iter_siblings<'a, ED: EdgeData>(&'a self, graph: &'a Graph<D, ED>) -> IterChain<&'a Node<D>, IterChildrenNodeExcept<'a, D, ED>> {
        IterChain::new(self.incoming_edges.iter().map(|&eid| {
            IterChildrenNodeExcept::new(self.id, &graph.nodes[graph.edges[eid].source_id].outgoing_edges, graph)
        }).collect())
    }

    pub fn iter_siblings_with_index<'a, ED: EdgeData>(&'a self, graph: &'a Graph<D, ED>, sibling_index: &'a SiblingIndex) -> IterNode<'a, D, ED> {
        IterNode::new(&sibling_index.siblings[self.id], graph)
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.kind as i32 + 1,
            "label": self.label,
            "data": self.data.to_dict()
        })
    }
}

