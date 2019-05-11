use serde_json::Value;
use data_structure::graph::graph::{Graph, EdgeData, NodeData, EmptyData};
use data_structure::graph::Node;

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum EdgeType {
    Unspecified = 0,
    Unknown = 1,
    URIProperty= 2,
    ObjectProperty = 3,
    DataProperty = 4
}

impl EdgeType {
    pub fn new(val: i8) -> EdgeType {
        match val {
            0 => EdgeType::Unspecified,
            1 => EdgeType::Unknown,
            2 => EdgeType::URIProperty,
            3 => EdgeType::ObjectProperty,
            4 => EdgeType::DataProperty,
            _ => panic!("Invalid type: {}", val)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge<D: EdgeData=EmptyData> {
    pub id: usize,
    pub kind: EdgeType,
    pub label: String,
    pub source_id: usize,
    pub target_id: usize,
    pub data: D
}

impl<D: EdgeData> Edge<D> {
    pub fn new(kind: EdgeType, label: String, source_id: usize, target_id: usize) -> Edge<D> {
        Edge {
            id: 182731897,
            kind,
            label,
            source_id,
            target_id,
            data: Default::default()
        }
    }

    pub fn new_with_data(kind: EdgeType, label: String, source_id: usize, target_id: usize, data: D) -> Edge<D> {
        Edge {
            id: 182731897,
            kind,
            label,
            source_id,
            target_id,
            data
        }
    }

    #[inline]
    pub fn get_source_node<'a, ND: NodeData>(&self, graph: &'a Graph<ND, D>) -> &'a Node<ND> {
        &graph.nodes[self.source_id]
    }

    #[inline]
    pub fn get_target_node<'a, ND: NodeData>(&self, graph: &'a Graph<ND, D>) -> &'a Node<ND> {
        &graph.nodes[self.target_id]
    }

    pub fn to_dict(&self) -> Value {
        json!({
            "id": self.id,
            "type": self.kind as i32,
            "label": self.label,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "data": self.data.to_dict()
        })
    }
}