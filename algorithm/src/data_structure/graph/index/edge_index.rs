use ndarray::prelude::Array2;
use data_structure::graph::*;

pub struct EdgeIndex {
    value: Array2<Vec<usize>>,
}

impl EdgeIndex {
    pub fn new<V: NodeData, E: EdgeData>(g: &Graph<V, E>) -> EdgeIndex {
        unimplemented!()
    }

    pub fn get_edge_between(&self, source_id: usize, target_id: usize) -> usize {
        unimplemented!()
    }
}