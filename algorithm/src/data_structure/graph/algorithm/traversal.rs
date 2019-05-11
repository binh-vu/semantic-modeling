use super::super::graph::{Graph, NodeData, EdgeData};
use super::super::node::Node;


pub struct DFSNode<'a, V: 'a + NodeData> {
    value: &'a Node<V>,
    should_remove: bool
}

/// Travel the graph (DFS) and apply `func` on each node. If the `func`
/// return false, we terminate the traversal
///
/// Return true and it terminate by exhausted, ortherwise false
pub fn dfs<'a, ND: NodeData, ED: EdgeData, F>(start_node: &'a Node<ND>, g: &'a Graph<ND, ED>, func: &mut F) -> bool
    where F: FnMut(&'a Node<ND>) -> bool {
    let mut stack = Vec::with_capacity(16);
    stack.push(start_node);
    while stack.len() > 0 {
        let node = stack.pop().unwrap();
        if !func(node) {
            return false;
        }

        for e in node.iter_outgoing_edges(g) {
            stack.push(e.get_target_node(g));
        }
    }

    return true;
}

pub fn dfs_full<'a, ND: NodeData, ED: EdgeData, F, F2>(start_node: &'a Node<ND>, g: &'a Graph<ND, ED>, begin_func: &mut F, end_func: &mut F2) -> bool
    where F: FnMut(&'a Node<ND>) -> bool, F2: FnMut(&'a Node<ND>) -> bool {

    let mut stack = Vec::with_capacity(16);
    stack.push(DFSNode { value: start_node, should_remove: false });

    while stack.len() > 0 {
        if !stack.last().unwrap().should_remove {
            // first encounter this node
            stack.last_mut().unwrap().should_remove = true;
            let node = stack.last().unwrap().value;
            if !begin_func(node) {
                return false;
            }

            for e in node.iter_outgoing_edges(g) {
                stack.push(DFSNode { value: e.get_target_node(g), should_remove: false });
            }
        } else {
            let node = stack.pop().unwrap();
            if !end_func(node.value) {
                return false;
            }
        }
    }

    return true;
}