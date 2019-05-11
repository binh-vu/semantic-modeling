use data_structure::graph::graph::{Graph, NodeData, EdgeData, EmptyData};
use data_structure::graph::Edge;
use data_structure::graph::Node;

pub struct IterEdge<'a, ND: 'a + NodeData=EmptyData, ED: 'a + EdgeData=EmptyData> {
    current_idx: usize,
    pub(super) edges: &'a Vec<usize>,
    pub(super) graph: &'a Graph<ND, ED>
}

pub struct IterNode<'a, ND: 'a + NodeData=EmptyData, ED: 'a + EdgeData=EmptyData> {
    current_idx: usize,
    pub(super) n_elements: usize,
    pub(super) nodes: &'a Vec<usize>,
    pub(super) graph: &'a Graph<ND, ED>
}

pub struct IterOwnedNode<'a, ND: 'a + NodeData=EmptyData, ED: 'a + EdgeData=EmptyData> {
    current_idx: usize,
    pub(super) n_elements: usize,
    pub(super) nodes: Vec<usize>,
    pub(super) graph: &'a Graph<ND, ED>
}

pub struct IterChildrenNode<'a, ND: 'a + NodeData=EmptyData, ED: 'a + EdgeData=EmptyData> {
    current_idx: usize,
    pub(super) n_elements: usize,
    pub(super) edges: &'a Vec<usize>,
    pub(super) graph: &'a Graph<ND, ED>
}

pub struct IterChildrenNodeExcept<'a, ND: 'a + NodeData=EmptyData, ED: 'a + EdgeData=EmptyData> {
    current_idx: usize,
    except_id: usize,
    pub(super) n_elements: usize,
    pub(super) edges: &'a Vec<usize>,
    pub(super) graph: &'a Graph<ND, ED>
}


pub struct IterChain<I, T: Iterator<Item=I>> {
    current_idx: usize,
    n_elements: usize,
    pub iterators: Vec<T>
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> IterEdge<'a, ND, ED> {
    pub fn new(edges: &'a Vec<usize>, graph: &'a Graph<ND, ED>) -> IterEdge<'a, ND, ED> {
        IterEdge {
            current_idx: 0,
            edges,
            graph
        }
    }

    pub fn empty(edges: &'a Vec<usize>, graph: &'a Graph<ND, ED>) -> IterEdge<'a, ND, ED> {
        IterEdge {
            current_idx: edges.len(),
            edges,
            graph
        }
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> IterNode<'a, ND, ED> {
    pub fn new(nodes: &'a Vec<usize>, graph: &'a Graph<ND, ED>) -> IterNode<'a, ND, ED> {
        IterNode {
            current_idx: 0,
            n_elements: nodes.len(),
            nodes,
            graph
        }
    }

    pub fn empty(nodes: &'a Vec<usize>, graph: &'a Graph<ND, ED>) -> IterNode<'a, ND, ED> {
        IterNode {
            current_idx: nodes.len(),
            n_elements: 0,
            nodes,
            graph
        }
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> IterOwnedNode<'a, ND, ED> {
    pub fn new(nodes: Vec<usize>, graph: &'a Graph<ND, ED>) -> IterOwnedNode<'a, ND, ED> {
        IterOwnedNode {
            current_idx: 0,
            n_elements: nodes.len(),
            nodes,
            graph
        }
    }

    pub fn empty(nodes: Vec<usize>, graph: &'a Graph<ND, ED>) -> IterOwnedNode<'a, ND, ED> {
        IterOwnedNode {
            current_idx: nodes.len(),
            n_elements: 0,
            nodes,
            graph
        }
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> IterChildrenNodeExcept<'a, ND, ED> {
    pub fn new(except_id: usize, edges: &'a Vec<usize>, graph: &'a Graph<ND, ED>) -> IterChildrenNodeExcept<'a, ND, ED> {
        IterChildrenNodeExcept {
            current_idx: 0,
            except_id,
            n_elements: edges.len(),
            edges,
            graph
        }
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> IterChildrenNode<'a, ND, ED> {
    pub fn new(edges: &'a Vec<usize>, graph: &'a Graph<ND, ED>) -> IterChildrenNode<'a, ND, ED> {
        IterChildrenNode {
            current_idx: 0,
            n_elements: edges.len(),
            edges,
            graph
        }
    }
}

impl<I, T: Iterator<Item=I>> IterChain<I, T> {
    pub fn new(iterators: Vec<T>) -> IterChain<I, T> {
        IterChain {
            current_idx: 0,
            n_elements: iterators.len(),
            iterators
        }
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> Iterator for IterEdge<'a, ND, ED> {
    type Item = &'a Edge<ED>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.current_idx < self.edges.len() {
            Some(&self.graph.edges[self.edges[self.current_idx]])
        } else {
            None
        };

        self.current_idx += 1;
        result
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> Iterator for IterNode<'a, ND, ED> {
    type Item = &'a Node<ND>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.current_idx < self.n_elements {
            Some(&self.graph.nodes[self.nodes[self.current_idx]])
        } else {
            None
        };

        self.current_idx += 1;
        result
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> Iterator for IterOwnedNode<'a, ND, ED> {
    type Item = &'a Node<ND>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.current_idx < self.n_elements {
            Some(&self.graph.nodes[self.nodes[self.current_idx]])
        } else {
            None
        };

        self.current_idx += 1;
        result
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> Iterator for IterChildrenNode<'a, ND, ED> {
    type Item = &'a Node<ND>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.current_idx < self.n_elements {
            Some(&self.graph.nodes[self.graph.edges[self.edges[self.current_idx]].target_id])
        } else {
            None
        };

        self.current_idx += 1;
        result
    }
}

impl<'a, ND: 'a + NodeData, ED: 'a + EdgeData> Iterator for IterChildrenNodeExcept<'a, ND, ED> {
    type Item = &'a Node<ND>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx < self.n_elements {
            let mut edge = &self.graph.edges[self.edges[self.current_idx]];
            if edge.target_id == self.except_id {
                self.current_idx += 1;
                if self.current_idx == self.n_elements {
                    return None;
                }

                edge = &self.graph.edges[self.edges[self.current_idx]];
            }

            self.current_idx += 1;
            return Some(&self.graph.nodes[edge.target_id]);
        } else {
            self.current_idx += 1;
            return None;
        }
    }
}

impl<I, T: Iterator<Item=I>> Iterator for IterChain<I, T> {
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx == self.n_elements {
            return None;
        }

        let mut result = self.iterators[self.current_idx].next();
        while self.current_idx < self.n_elements && result.is_none() {
            self.current_idx += 1;
            result = self.iterators[self.current_idx].next();
        }

        return result;
    }
}