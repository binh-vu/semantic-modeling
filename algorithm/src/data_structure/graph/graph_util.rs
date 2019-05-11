use super::graph::{Graph, NodeData, EdgeData};
use super::node::{Node, NodeType};
use super::edge::EdgeType;
use std::fs::{File, remove_file};
use std::io::prelude::*;
use std::path::Path;
use string::auto_wrap;
use data_structure::graph::Edge;
use fnv::FnvHashSet;
use std::process::Command;
use regex::Regex;
use std::collections::HashMap;

fn tree_hashing<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>, n: &Node<ND>, incoming_edge_id: i32, visited_index: &mut Vec<i32>) -> String {
    if visited_index[n.id] == incoming_edge_id {
        // re-visit via same link, cycle detection
        return "".to_owned();
    }

    visited_index[n.id] = incoming_edge_id;
    if n.n_outgoing_edges == 0 {
        return n.label.clone();
    }

    let mut children_texts = Vec::new();
    for e in n.iter_outgoing_edges(g) {
        let result = tree_hashing(g, &g.nodes[e.target_id], e.id as i32, visited_index);
        children_texts.push(format!("{}-{}", e.label, result));
    }

    children_texts.sort();
    return format!("({}:{})", n.label, children_texts.join(","));
}

pub fn get_acyclic_consistent_unique_hashing<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>) -> String {
    // we don't guarantee that it will return consistent result if graph has cycle
    let mut roots = Vec::new();
    let mut visited_index = vec![-2; g.n_nodes];

    for n in g.iter_nodes() {
        if n.n_incoming_edges == 0 {
            roots.push(tree_hashing(g, n, -1, &mut visited_index))
        }
    }

    return roots.join(",");
}

pub fn graph2dot<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>, foutput: &str, max_text_width: Option<u32>) {
    let max_text_width = match max_text_width {
        None => 15,
        Some(x) => x
    };

    let mut f = File::create(foutput).unwrap();
    let s = format!("digraph n0 {{ fontcolor=\"blue\"\n remincross=\"true\"\n label=\"{}\"\n", g.id);
    f.write(s.as_bytes()).unwrap();

    for n in g.iter_nodes() {
        let label = auto_wrap(&format!("N{:03}: {}", n.id, n.label), max_text_width, None, true);
        if n.is_class_node() {
            f.write(format!("\"{}\"[style=\"filled\",color=\"white\",fillcolor=\"lightgray\",label=\"{}\"];\n", n.id, label).as_bytes()).unwrap();
        } else {
            f.write(format!("\"{}\"[shape=\"plaintext\",style=\"filled\",fillcolor=\"gold\",label=\"{}\"];\n", n.id, label).as_bytes()).unwrap();
        }
    }

    for e in g.iter_edges() {
        let label = auto_wrap(&format!("L{:03}: {}", e.id, e.label), max_text_width, None, true);
        f.write(format!("\"{}\" -> \"{}\"[color=\"brown\",fontcolor=\"black\",label=\"{}\"];\n", e.source_id, e.target_id, label).as_bytes()).unwrap();
    }

    f.write(b"}\n").unwrap();
}

pub fn graph2pdf<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>, foutput: &str, max_text_width: Option<u32>) {
    let tmp_file = format!("{}.tmp", foutput);

    graph2dot(g, &tmp_file, max_text_width);
    let output = Command::new("dot")
        .arg("-Tpdf")
        .arg(&tmp_file)
        .arg(&format!("-o{}", foutput))
        .output()
        .expect("Fail to execute graphviz process");
    assert!(output.status.success(), "Fail to execute graphviz process. Reason: {}", String::from_utf8(output.stderr).unwrap());

    remove_file(tmp_file).unwrap();
}

pub fn foreach_descendant_nodes<'a, ND: NodeData, ED: EdgeData, F>(g: &'a Graph<ND, ED>, start_node: &'a Node<ND>, mut func: F)
    where F: FnMut(&'a Node<ND>) {
    let mut visited_nodes: Vec<bool> = vec![false; g.n_nodes];

    for e in start_node.iter_outgoing_edges(g) {
        if visited_nodes[e.target_id] {
            continue;
        }

        visited_nodes[e.target_id] = true;
        let target = e.get_target_node(g);
        func(target);

        if target.n_outgoing_edges > 0 {
            real_foreach_descendant_nodes(g, target, &mut visited_nodes, &mut func);
        }
    }
}

fn real_foreach_descendant_nodes<'a, ND: NodeData, ED: EdgeData, F>(g: &'a Graph<ND, ED>, start_node: &'a Node<ND>, visited_nodes: &mut Vec<bool>, func: &mut F)
    where F: FnMut(&'a Node<ND>) {
    for e in start_node.iter_outgoing_edges(g) {
        if visited_nodes[e.target_id] {
            continue;
        }

        visited_nodes[e.target_id] = true;
        let target = e.get_target_node(g);
        func(target);

        if target.n_outgoing_edges > 0 {
            real_foreach_descendant_nodes(g, target, visited_nodes, func);
        }
    }
}

pub fn get_descendant_edges<'a, ND: NodeData, ED: EdgeData>(g: &'a Graph<ND, ED>, edge: &'a Edge<ED>) -> Vec<usize> {
    let mut descendant_edges: FnvHashSet<usize> = Default::default();
    let target_node = edge.get_target_node(g);
    for e in target_node.iter_outgoing_edges(g) {
        if !descendant_edges.contains(&e.id) {
            descendant_edges.insert(e.id);
            get_descendant_edges_real(g, e, &mut descendant_edges);
        }
    }

    descendant_edges.into_iter().collect()
}

pub fn get_descendant_edges_real<'a, ND: NodeData, ED: EdgeData>(g: &'a Graph<ND, ED>, edge: &'a Edge<ED>, descendant_edges: &mut FnvHashSet<usize>) {
    let target_node = edge.get_target_node(g);
    for e in target_node.iter_outgoing_edges(g) {
        if !descendant_edges.contains(&e.id) {
            descendant_edges.insert(e.id);
            get_descendant_edges_real(g, e, descendant_edges);
        }
    }
}


/// Take a sequence of edges in form of: [<source_id>--<edge_lbl>--<target_id>, ...], convert
/// it into a graph.
///
/// Note that you can append `::d` or `::c` to set the node type to be class node or data node
/// default is class node (`::c`)
///
/// Sometime the graph can contains multiple nodes which have same label, and so an number has
/// been added to the back, so we have unique id for the node. Therefore, a node id with
/// number on the tail will be remove!
pub fn quick_graph<T: AsRef<str>>(edges: &[T]) -> (Graph, HashMap<String, usize>) {
    let mut g = Graph::new("quick_graph".to_owned(), true, true, true);
    let mut idmap: HashMap<String, usize> = Default::default();
    let type_regex = Regex::new(r"(?:(?:(.+)::(d|c))|(.+))$").unwrap();
    let name_regex = Regex::new(r"(.*[a-zA-Z])(\d+)?$").unwrap();

    for e in edges {
        let mut ed = e.as_ref().split("--");
        let source_captures = type_regex.captures(ed.next().unwrap()).unwrap();
        let (source_id, source_type) = match source_captures.get(2) {
            None => (source_captures.get(3).unwrap().as_str(), NodeType::ClassNode),
            Some(x) => {
                if x.as_str() == "d" {
                    (source_captures.get(1).unwrap().as_str(), NodeType::DataNode)
                } else {
                    (source_captures.get(1).unwrap().as_str(), NodeType::ClassNode)
                }
            },
        };

        if !idmap.contains_key(source_id) {
            let label = name_regex.captures(source_id).unwrap().get(1).unwrap().as_str().to_owned();
            idmap.insert(source_id.to_owned(), g.add_node(Node::new(source_type, label)));
        }

        if let Some(edge_label) = ed.next() {
            let target_captures = type_regex.captures(ed.next().unwrap()).unwrap();
            let (target_id, target_type) = match target_captures.get(2) {
                None => (target_captures.get(3).unwrap().as_str(), NodeType::ClassNode),
                Some(x) => {
                    if x.as_str() == "d" {
                        (target_captures.get(1).unwrap().as_str(), NodeType::DataNode)
                    } else {
                        (target_captures.get(1).unwrap().as_str(), NodeType::ClassNode)
                    }
                },
            };

            if !idmap.contains_key(target_id) {
                let label = name_regex.captures(target_id).unwrap().get(1).unwrap().as_str().to_owned();
                idmap.insert(target_id.to_owned(), g.add_node(Node::new(target_type, label)));
            }

            g.add_edge(Edge::new(EdgeType::Unspecified, edge_label.to_owned(), idmap[source_id], idmap[target_id]));
        }
    }

    (g, idmap)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn test_quick_graph() {
        let mut corr_g = Graph::new("quick_graph".to_owned(), true, true, true);
        corr_g.add_node(Node::new(NodeType::ClassNode, "A".to_owned()));
        assert_eq!(quick_graph(&["A1"]).0, corr_g);
        corr_g.add_node(Node::new(NodeType::ClassNode, "B".to_owned()));
        corr_g.add_edge(Edge::new(EdgeType::Unspecified, "x".to_owned(), 0, 1));
        assert_eq!(quick_graph(&["A1--x--B1"]).0, corr_g);

        let g = quick_graph(&["A1::c--x--B1::c", "B1--y--C::d"]).0;
        corr_g.add_node(Node::new(NodeType::DataNode, "C".to_owned()));
        corr_g.add_edge(Edge::new(EdgeType::Unspecified, "y".to_owned(), 1, 2));
        assert_eq!(g, corr_g);


    }
}