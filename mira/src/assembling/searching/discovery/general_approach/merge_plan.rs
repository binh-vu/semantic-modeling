use super::graph_explorer::GraphExplorer;
use std::collections::HashSet;

use algorithm::data_structure::graph::*;

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IntegrationPoint {
    // all the id is local (not the final id we get after merge 2 graph together)
    // either source_id or target_id is in part_a/part_b, the remain id is in part_merge
    // if is_incoming_edge = true, then target in part_a/part_b, source_id in part_merge
    // reverse if otherwise.
    pub is_incoming_edge: bool,
    pub edge_id: usize,
    pub source_id: usize,
    pub target_id: usize
}

impl IntegrationPoint {
    #[inline]
    pub fn new(is_incoming_edge: bool, source_id: usize, edge_id: usize, target_id: usize) -> IntegrationPoint {
        IntegrationPoint {
            is_incoming_edge,
            edge_id,
            source_id,
            target_id
        }
    }

}

#[derive(Serialize, Deserialize, Eq, PartialEq)]
pub struct MergePlan {
    pub int_tree: Graph,
    pub int_a: IntegrationPoint,
    pub int_b: IntegrationPoint
}

impl MergePlan {
    #[inline]
    pub fn new(int_tree: Graph, int_a: IntegrationPoint, int_b: IntegrationPoint) -> MergePlan {
        MergePlan {
            int_tree, int_a, int_b
        }
    }
}

fn add_merge_path(g_merge: &mut Graph, explorer: &GraphExplorer, x_n_id: usize) -> (usize, usize) {
    // A merge path is like this: A -- X_1 -- X_2 -- ... -- X_N -- B where N can vary from 0 to inf.
    // Input:
    //    * a graph where the merge path will be added to
    //    * a graph explorer which contains the merge_path, A is always in this graph.
    //    * id of X_N in a graph explorer, and add all path from X_N downback to X_1
    // Output:
    //    if N > 0 or N < 0: id of link A -- X_1 and id of X_1 in `g_merge` so that they can complete the merge path by adding A -- X_1
    //    if N == 0: (x, x) instead of (-1, -1) like in python code, since we know they won't be used
    //        so x would be some number that will cause the program terminate if use incorrectly
    //        choose x = 100_000_000 + g_merge.n_nodes + explorer.n_nodes

    let x_i_n_hop = explorer.get_hop_by_id(x_n_id);
    if x_i_n_hop == 0 {
        return (100_000_000 + g_merge.n_nodes + explorer.n_nodes, 100_000_000 + g_merge.n_nodes + explorer.n_nodes);
    }

    let mut x_i = explorer.get_node_by_id(x_n_id);
    let mut y_i_id = g_merge.add_node(Node::new(x_i.kind, x_i.label.clone()));

    if x_i_n_hop > 0 {
        // -1 so that we only go to X_1 not A
        for _i in 0..(x_i_n_hop - 1) {
            // must have incoming link ids
            let edge = explorer.graph.get_edge_by_id(x_i.incoming_edges[0]);
            let x_i_1 = explorer.graph.get_node_by_id(edge.source_id);
            let y_i_1_id = g_merge.add_node(Node::new(x_i_1.kind, x_i_1.label.clone()));

            g_merge.add_edge(Edge::new(edge.kind, edge.label.clone(), y_i_1_id, y_i_id));
            x_i = x_i_1;
            y_i_id = y_i_1_id;
        }

        (x_i.incoming_edges[0], y_i_id)
    } else {
        for _i in 0..(-x_i_n_hop - 1) {
            let edge = explorer.graph.get_edge_by_id(x_i.outgoing_edges[0]);
            let x_i_1 = explorer.graph.get_node_by_id(edge.target_id);
            let y_i_1_id = g_merge.add_node(Node::new(x_i_1.kind, x_i_1.label.clone()));

            g_merge.add_edge(Edge::new(edge.kind, edge.label.clone(), y_i_id, y_i_1_id));
            x_i = x_i_1;
            y_i_id = y_i_1_id;
        }

        (x_i.outgoing_edges[0], y_i_id)
    }
}

pub fn combine_merge_path(g_merge_a: &Graph, g_merge_b: &Graph, x_n_id: usize, y_n_id: usize) -> Graph {
    // Combine 2 merge paths together, each of them are represented by a graph of linear chain X_1 -- X_2 -- ... -- X_N
    // and Y_1 -- Y_2 -- ... -- Y_N. The points to connect 2 chains are X_N and Y_N, 
    // output: X_1 -- X_2 -- ... X_N -- Y_N_1 -- ... -- Y_1
    let mut g_merge = g_merge_a.clone();
    let mut id_map = Vec::with_capacity(g_merge_b.n_nodes);

    for n in g_merge_b.iter_nodes() {
        // exploit the fact that ids are assigned continuously
        id_map.push(if n.id != y_n_id {
            g_merge.add_node(Node::new(n.kind, n.label.clone()))
        } else {
            x_n_id
        });
    }

    for e in g_merge_b.iter_edges() {
        g_merge.add_edge(Edge::new(e.kind, e.label.clone(), id_map[e.source_id], id_map[e.target_id]));
    }

    g_merge
}

pub fn make_plan4case23_subfunc(_tree_a: &Graph, tree_a_search: &GraphExplorer, root_b: &Node) -> Vec<MergePlan> {
    // Generate plan that merge tree b into tree a (A --> B)
    let mut plans = Vec::new();

    for b in tree_a_search.iter_gt0hop_nodes_by_label(&root_b.label) {
        let edge = b.first_incoming_edge(tree_a_search).unwrap();
        let x_n = edge.get_source_node(tree_a_search);
        let x_n_hop = tree_a_search.get_hop_by_id(x_n.id);
        let mut g_merge = Graph::new("".to_owned(), false, false, false);

        if x_n_hop == 0 {
            g_merge.add_node(Node::new(x_n.kind, x_n.label.clone())); // A' = 0
            g_merge.add_node(Node::new(b.kind, b.label.clone())); // B' = 1
            g_merge.add_edge(Edge::new(edge.kind, edge.label.clone(), 0, 1)); // (A' -> B')

            let int_a = IntegrationPoint::new(false, x_n.id, 0, 1); // source=A, target=B'
            let int_b = IntegrationPoint::new(true, 0, 0, root_b.id); // source=A', target=root_b
            plans.push(MergePlan::new(g_merge, int_a, int_b));
        } else {
            let (link_a_id, x_1_id_prime) = add_merge_path(&mut g_merge, tree_a_search, x_n.id);
            let edge_a = tree_a_search.graph.get_edge_by_id(link_a_id as usize);
            let node_a = edge_a.get_source_node(&tree_a_search.graph);

            let b_id_prime = g_merge.add_node(Node::new(b.kind, b.label.clone()));
            let a_id_prime = g_merge.add_node(Node::new(node_a.kind, node_a.label.clone()));

            let edge_a_id_prime = g_merge.add_edge(Edge::new(edge_a.kind, edge_a.label.clone(), a_id_prime, x_1_id_prime));
            let edge_b_id_prime = g_merge.add_edge(Edge::new(edge.kind, edge.label.clone(), 0, b_id_prime)); // X'_n = 0
            
            let int_a = IntegrationPoint::new(false, node_a.id, edge_a_id_prime, x_1_id_prime);
            let int_b = IntegrationPoint::new(true, 0, edge_b_id_prime, root_b.id);
            plans.push(MergePlan::new(g_merge, int_a, int_b));
        }
    }

    plans
}

pub fn make_plan4case23_specialcase(tree_a: &Graph, tree_a_search: &GraphExplorer, root_b: &Node, tree_b_search: &GraphExplorer) -> Vec<MergePlan> {
    let mut plans = Vec::new();

    // note that node in Graph share same id like node in GraphExplorer
    for link in tree_b_search.get_node_by_id(root_b.id).iter_incoming_edges(tree_b_search) {
        let p_b = link.get_source_node(tree_b_search);
        for x_n in tree_a_search.iter_gt0hop_nodes_by_label(&p_b.label) {
            let mut g_merge = Graph::new("".to_owned(), false, false, false);
            let (link_a_id, x_1_id_prime) = add_merge_path(&mut g_merge, tree_a_search, x_n.id);
            let link_a = tree_a_search.get_edge_by_id(link_a_id);
            let node_a = link_a.get_source_node(tree_a_search);

            let a_id_prime = g_merge.add_node(Node::new(node_a.kind, node_a.label.clone()));
            let b_id_prime = g_merge.add_node(Node::new(root_b.kind, root_b.label.clone()));

            let link_a_id_prime = g_merge.add_edge(Edge::new(link_a.kind, link_a.label.clone(), a_id_prime, x_1_id_prime));
            let link_b_id_prime = g_merge.add_edge(Edge::new(link.kind, link.label.clone(), 0, b_id_prime));
            let int_a = IntegrationPoint::new(false, node_a.id, link_a_id_prime, x_1_id_prime);
            let int_b = IntegrationPoint::new(true, 0, link_b_id_prime, root_b.id);

            plans.push(MergePlan::new(g_merge, int_a, int_b));
        }

        for x_n in tree_a.iter_nodes_by_label(&p_b.label) {
            let mut g_merge = Graph::new("".to_owned(), false, false, false);
            g_merge.add_node(Node::new(p_b.kind, p_b.label.clone()));
            g_merge.add_node(Node::new(root_b.kind, root_b.label.clone()));
            g_merge.add_edge(Edge::new(link.kind, link.label.clone(), 0, 1));
            let int_a = IntegrationPoint::new(false, x_n.id, 0, 1);
            let int_b = IntegrationPoint::new(true, 0, 0, root_b.id);

            plans.push(MergePlan::new(g_merge, int_a, int_b));
        }
    }

    plans
}

pub fn make_plan4case23(tree_a: &Graph, tree_b: &Graph, tree_a_search: &GraphExplorer, tree_b_search: &GraphExplorer) -> Vec<MergePlan> {
    let root_a = tree_a.get_first_root_node().unwrap();
    let root_b = tree_b.get_first_root_node().unwrap();

    if root_b.is_data_node() && root_a.is_class_node() {
        return make_plan4case23_specialcase(tree_a, tree_a_search, &root_b, tree_b_search);
    } else if root_a.is_data_node() && root_b.is_class_node() {
        return make_plan4case23_specialcase(tree_b, tree_b_search, &root_a, tree_a_search);
    }

    let mut plans = make_plan4case23_subfunc(tree_a, tree_a_search, &root_b);
    plans.append(&mut make_plan4case23_subfunc(tree_b, tree_b_search, &root_a));
    plans
}

pub fn make_merge_plans4case1(_tree_a: &Graph, _tree_b: &Graph, tree_a_search: &GraphExplorer, tree_b_search: &GraphExplorer) -> Vec<MergePlan> {
    let mut a_ancestors: HashSet<&String> = Default::default();
    let mut common_ancestors: Vec<&String> = Default::default();

    // get common ancestors
    for n in tree_a_search.iter_lt0hop_nodes() {
        a_ancestors.insert(&n.label);
    }

    for n in tree_b_search.iter_lt0hop_nodes() {
        if a_ancestors.contains(&n.label) {
            common_ancestors.push(&n.label);
        }
    }

    let mut plans = Vec::new();

    // make plans from those ancestors
    for common_ancestor in common_ancestors.iter() {
        let mut merge_plan_a = Vec::new();
        let mut merge_plan_b = Vec::new();

        // make plan from a => common ancestor
        for n in tree_a_search.iter_lt0hop_nodes_by_label(common_ancestor) {
            let mut g_merge = Graph::new("".to_owned(), false, false, false);
            let (link_a_id, x_1_id_prime) = add_merge_path(&mut g_merge, tree_a_search, n.id);
            merge_plan_a.push((g_merge, link_a_id, x_1_id_prime));
        }

        for n in tree_b_search.iter_lt0hop_nodes_by_label(common_ancestor) {
            let mut g_merge = Graph::new("".to_owned(), false, false, false);
            let (link_b_id, y_1_id_prime) = add_merge_path(&mut g_merge, tree_b_search, n.id);
            merge_plan_b.push((g_merge, link_b_id, y_1_id_prime));
        }

        for &(ref g_merge_a, link_a_id, x_1_id_prime) in merge_plan_a.iter() {
            let edge_a = tree_a_search.get_edge_by_id(link_a_id);
            let node_a = edge_a.get_target_node(&tree_a_search.graph);

            for &(ref g_merge_b, link_b_id, y_1_id_prime) in merge_plan_b.iter() {
                let edge_b = tree_b_search.get_edge_by_id(link_b_id);
                let node_b = edge_b.get_target_node(&tree_b_search.graph);

                // note that X_n_prime & Y_n_prime always 0 due to the fact that id is assigned continuously
                let y_1_id_prime = if y_1_id_prime != 0 {
                    // NOTE: we add link B - Y1' to g_merge, but because when we add Y1 to g_merge, it has different id
                    // the new id is the old id shifted by |g_merge_a.n_nodes| - 1
                    y_1_id_prime + g_merge_a.n_nodes - 1
                } else {
                    // there is a case where g_merge_b have only one node, then y_1_id_prime is actually node 0, which is
                    // removed by merging into X_N in g_merge_a
                    y_1_id_prime
                };

                let mut g_merge = combine_merge_path(&g_merge_a, &g_merge_b, 0, 0);
                let a_id_prime = g_merge.add_node(Node::new(node_a.kind, node_a.label.clone()));
                let b_id_prime = g_merge.add_node(Node::new(node_b.kind, node_b.label.clone()));

                let link_a_id_prime = g_merge.add_edge(Edge::new(edge_a.kind, edge_a.label.clone(), x_1_id_prime, a_id_prime));
                let link_b_id_prime = g_merge.add_edge(Edge::new(edge_b.kind, edge_b.label.clone(), y_1_id_prime, b_id_prime));

                plans.push(MergePlan::new(
                    g_merge, 
                    IntegrationPoint::new(true, x_1_id_prime, link_a_id_prime, node_a.id),
                    IntegrationPoint::new(true, y_1_id_prime, link_b_id_prime, node_b.id)
                ));
            }
        }
    }

    plans
}

pub fn make_merge_plans(tree_a: &Graph, tree_b: &Graph, tree_a_search: &GraphExplorer, tree_b_search: &GraphExplorer) -> Vec<MergePlan> {
    let mut plans = make_merge_plans4case1(tree_a, tree_b, tree_a_search, tree_b_search);
    plans.append(&mut make_plan4case23(tree_a, tree_b, tree_a_search, tree_b_search));
    plans
}