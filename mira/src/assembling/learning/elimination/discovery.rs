use assembling::models::annotator::Annotator;
use rdb2rdf::models::semantic_model::SemanticModel;
use fnv::FnvHashSet;
use fnv::FnvHashMap;
use im::HashSet as IHashSet;
use im::OrdSet as IOrdSet;
use std::collections::HashSet;
use algorithm::prelude::*;
use assembling::searching::beam_search::SearchNode;
use assembling::searching::discovery::general_approach::*;
use assembling::searching::discovery::constraint_space::*;
use assembling::auto_label;
use evaluation_metrics::semantic_modeling::DataNodeMode;
use settings::Settings;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use utils::set_has;

pub fn cascade_remove_downward(nid: usize, graph: &Graph, delete_node_ids: &mut FnvHashSet<usize>, delete_edge_ids: &FnvHashSet<usize>, new_delete_edges: &mut Vec<usize>) {
    delete_node_ids.insert(nid);
    for e in graph.get_node_by_id(nid).iter_outgoing_edges(&graph) {
        new_delete_edges.push(e.id);
        cascade_remove_downward(e.target_id, graph, delete_node_ids, delete_edge_ids, new_delete_edges);
    }
}


pub fn cascade_remove<'a>(graph: &Graph, mut delete_node_ids: FnvHashSet<usize>, mut delete_edge_ids: FnvHashSet<usize>) -> Graph {
    let mut rewire_idmap: FnvHashMap<usize, usize> = Default::default();

    // propagate the deletion downward
    let mut new_delete_edges = Vec::new();
    for &eid in &delete_edge_ids {
        cascade_remove_downward(graph.get_edge_by_id(eid).target_id, graph, &mut delete_node_ids, &delete_edge_ids, &mut new_delete_edges);
    }
    delete_edge_ids.extend(new_delete_edges.into_iter());

    // propagate the deletion upward
    loop {
        let mut new_delete_nodes = Vec::new();
        for &nid in delete_node_ids.iter() {
            let node = graph.get_node_by_id(nid);
            match node.first_incoming_edge(&graph) {
                None => (),
                Some(pe) => {
                    // if its parent edge has not been included
                    if !delete_edge_ids.contains(&pe.id) {
                        delete_edge_ids.insert(pe.id);
                    }

                    // examine if we can delete the parent node as well
                    if !delete_node_ids.contains(&pe.source_id) {
                        let parent_node = pe.get_source_node(&graph);
                        if parent_node.outgoing_edges.iter().all(|eid| delete_edge_ids.contains(eid)) {
                            new_delete_nodes.push(pe.source_id);
                        }
                    }
                }
            }
        }

        if new_delete_nodes.len() == 0 {
            break;
        } else {
            delete_node_ids.extend(new_delete_nodes.into_iter());
        }
    }

    let mut new_g = Graph::new_like(graph);
    for n in graph.iter_nodes() {
        if !delete_node_ids.contains(&n.id) {
            rewire_idmap.insert(n.id, new_g.add_node(Node::new(n.kind, n.label.clone())));
        }
    }

    for e in graph.iter_edges() {
        if !delete_edge_ids.contains(&e.id) {
            new_g.add_edge(Edge::new(e.kind, e.label.clone(), rewire_idmap[&e.source_id], rewire_idmap[&e.target_id]));
        }
    }

    new_g
}

#[inline]
pub fn remove_no_candidate_stypes<'a>(sm: &SemanticModel) -> Graph {
    let mut deleted_node_ids: FnvHashSet<usize> = Default::default();
    let mut deleted_edge_ids: FnvHashSet<usize> = Default::default();

    for attr in sm.attrs.iter() {
        let data_node = sm.graph.get_node_by_id(attr.id);
        let e = data_node.first_incoming_edge(&sm.graph).unwrap();
        let parent = e.get_source_node(&sm.graph);

        let mut deleted_nodes = true;
        for stype in attr.semantic_types.iter() {
            if stype.class_uri == parent.label && stype.predicate == e.label {
                deleted_nodes = false;
            }
        }

        if deleted_nodes {
            deleted_node_ids.insert(data_node.id);
            deleted_edge_ids.insert(e.id);
        }
    }

    cascade_remove(&sm.graph, deleted_node_ids, deleted_edge_ids)
}

pub fn get_started_eliminated_1nodes<'a>(annotator: &'a Annotator, sm: &SemanticModel) -> Vec<(Graph, Vec<String>)> {
    // the started nodes will be the graph, which discard 1 or more attributes
    // start with 1 attributes first
    let new_g = remove_no_candidate_stypes(sm);

    let mut next_states = Vec::new();

    // now loop through one attributes and remove this attribute one by one
    for dnode in new_g.iter_data_nodes() {
        // create new state without this dnode
        let mut deleted_node_ids: FnvHashSet<usize> = Default::default();
        deleted_node_ids.insert(dnode.id);
        let new_g_without_this_node = cascade_remove(&new_g, deleted_node_ids, Default::default());

        if new_g_without_this_node.n_nodes == 0 {
            continue;
        }

        next_states.push((new_g_without_this_node, vec![dnode.label.clone()]));
    }

    return next_states;
}

pub fn get_started_eliminated_2nodes<'a>(annotator: &'a Annotator, sm: &SemanticModel) -> Vec<(Graph, Vec<String>)> {
    let new_g = remove_no_candidate_stypes(sm);
    let mut next_states = Vec::new();
    let mut visited_nodes = HashSet::<(String, String)>::new();

    for dnode in new_g.iter_data_nodes() {
        // create new state without this node
        let mut remained_terminals: IHashSet<String> = Default::default();
        remained_terminals.insert(dnode.label.clone());

        let mut deleted_node_ids: FnvHashSet<usize> = Default::default();
        deleted_node_ids.insert(dnode.id);

        let new_g1node = cascade_remove(&new_g, deleted_node_ids, Default::default());

        // loop through each stype, if there are a semantic type match in this graph, we will remove that node too
        let attr = sm.get_attr_by_label(&dnode.label);
        let n_next_states = next_states.len();

        for stype in attr.semantic_types.iter() {
            for potential_node in new_g1node.iter_nodes_by_label(&stype.class_uri) {
                for potential_e in potential_node.iter_outgoing_edges(&new_g1node) {
                    let potential_target = potential_e.get_target_node(&new_g1node);

                    if potential_target.id != dnode.id && potential_e.label == stype.predicate && potential_target.is_data_node() {
                        if !set_has(&visited_nodes,&dnode.label, &potential_target.label) {
                            let mut deleted_node_ids: FnvHashSet<usize> = Default::default();
                            deleted_node_ids.insert(potential_e.target_id);
                            let new_g2node = cascade_remove(&new_g1node, deleted_node_ids, Default::default());

                            next_states.push((new_g2node, vec![dnode.label.clone(), potential_target.label.clone()]));

                            visited_nodes.insert((dnode.label.clone(), potential_target.label.clone()));
                            visited_nodes.insert((potential_target.label.clone(), dnode.label.clone()));
                        }
                    }
                }
            }
        }

        if n_next_states == next_states.len() {
            next_states.push((new_g1node, vec![dnode.label.clone()]));
        }
    }

    next_states
}

pub fn convert_to_general_discovery_nodes(sm: &SemanticModel, started_graphs: Vec<(Graph, Vec<String>)>, args: &mut GeneralDiscoveryArgs) -> Vec<GeneralDiscoveryNode> {
    let singularity = GraphDiscovery::init(&sm.attrs, args);

    started_graphs.into_iter()
        .map(|(g, dnodes)| {
            let g_explore = args.graph_explorer_builder.build(&g);
            let remained_terminals = dnodes.into_iter().collect::<IHashSet<String>>();
            let graph_discovery = singularity.replace_current_state(remained_terminals, g_explore, g, 1.0);

            SearchNode::new(graph_discovery.get_current_id(), graph_discovery.current_score, graph_discovery)
        })
        .collect::<Vec<_>>()
}

pub fn convert_to_constraint_nodes(sm: &SemanticModel, int_graph: &IntGraph, started_graphs: Vec<(Graph, Vec<String>)>) -> Vec<IntTreeSearchNode> {
    let max_permutation = Settings::get_instance().learning.max_permutation;

    started_graphs.into_iter()
        .map(|(g, attrs)| {
//            let bijections = auto_label::alignment::align_graph(&sm.graph, &g, DataNodeMode::NoTouch, max_permutation)
//                .into_iter()
//                .map(|r| Bijection::from_pairs(r.3.prime2x.iter().map(|&x| x as usize).collect::<Vec<_>>()))
//                .collect::<Vec<_>>();
//            let bijections = vec![Bijection::from_pairs(auto_label::alignment::align_graph_one(&sm.graph, &g, DataNodeMode::NoTouch, max_permutation).unwrap().3
//                                                            .prime2x.iter()
//                                                            .map(|&x| x as usize).collect::<Vec<_>>())];
            let bijections = vec![get_bijection(&sm, &g, int_graph)];

            let node_data = IntTreeSearchNodeData::new(g, attrs.into_iter().collect::<IOrdSet<String>>(), bijections);
            SearchNode::new(node_data.get_id(), 1.0, node_data)
        })
        .collect::<Vec<_>>()
}


fn get_bijection(sm: &SemanticModel, g: &Graph, int_graph: &IntGraph) -> Bijection {
    let no_node: usize = 999_998;
    // align between sm and g first
    let mut sm2g: FnvHashMap<usize, usize> = FnvHashMap::default();
    let mut stack = Vec::new();

    for n in sm.graph.iter_data_nodes() {
        match g.iter_nodes_by_label(&n.label).next() {
            None => {
                sm2g.insert(n.id, no_node);
            },
            Some(pn) => {
                sm2g.insert(n.id, pn.id);
                stack.push((n, pn));
            }
        }
    }

    while stack.len() > 0 {
        let (n, pn) = stack.pop().unwrap();
        for e in n.iter_incoming_edges(&sm.graph) {
            let sn = e.get_source_node(&sm.graph);

            for pe in pn.iter_incoming_edges(&g) {
                if pe.label == e.label {
                    let psn = pe.get_source_node(&g);
                    if sn.label == psn.label && !sm2g.contains_key(&sn.id) {
                        // this is a match
                        sm2g.insert(sn.id, psn.id);
                        stack.push((sn, psn));
                    }
                }
            }

            if !sm2g.contains_key(&sn.id) {
                sm2g.insert(sn.id, no_node);
            }
        }
    }

    for n in sm.graph.iter_nodes() {
        if !sm2g.contains_key(&n.id) {
            sm2g.insert(n.id, no_node);
        }
    }

    // align between int_graph with sm, then with g
    let mut prime2x = vec![999_999; g.n_nodes];
    for node in int_graph.graph.iter_nodes() {
        if let Some(x_prime) = node.data.original_ids.get(&sm.id) {
            let gxprime = *sm2g.get(x_prime).unwrap();
            if gxprime != no_node {
                prime2x[gxprime] = node.id;
            }
        }
    }

    let bijection = Bijection::from_pairs(prime2x);

    bijection
}


#[cfg(test)]
mod tests {
    use assembling::tests::tests::*;
    use super::*;

    #[test]
    pub fn test_get_started_eliminated_2nodes() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let sm = input.get_train_sms()[0];

        let started_nodes = get_started_eliminated_2nodes(&input.get_annotator(), sm);
        assert_eq!(started_nodes.len(), 90);
        assert_eq!(started_nodes[..10].iter()
                       .map(|x| x.1.iter().map(|y| y.as_str()).collect::<Vec<_>>())
                       .collect::<Vec<_>>(), vec![
            vec!["BirthDateLatest", "DeathDateLatest"],
            vec!["BirthDateLatest", "DateEnd"],
            vec!["BirthDateLatest", "BirthDateEarliest"],
            vec!["BirthDateLatest", "DeathDateEarliest"],
            vec!["BirthDateLatest", "DateBegin"],
            vec!["BirthDateLatest", "Begin Date"],
            vec!["BirthDateLatest", "End Date"],
            vec!["BirthDateLatest", "Dated"],
            vec!["Attribution", "Dimensions"],
            vec!["Attribution", "Medium"]
        ]);
    }

    #[test]
    pub fn test_get_bijection() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let sm = input.get_train_sms()[0];
        let int_graph = IntGraph::new(&input.get_train_sms());

        let started_nodes = get_started_eliminated_2nodes(&input.get_annotator(), sm);
        for (g, _) in started_nodes.iter() {
            let bijection = get_bijection(&sm, g, &int_graph);
            for n in g.iter_nodes() {
                let x = int_graph.graph.get_node_by_id(bijection.to_x(n.id));
                assert!(n.n_incoming_edges <= 1);

                if let Some(e) = n.iter_incoming_edges(g).next() {
                    let mut flag = false;
                    for pe in x.iter_incoming_edges(&int_graph.graph) {
                        if pe.source_id == bijection.to_x(e.source_id) {
                            assert_eq!(pe.label, e.label);
                            flag = true;
                        }
                    }
                    assert!(flag);
                }

                if x.is_data_node() {
                    assert!(n.is_data_node(), "x = {} - x_prime = {}", x.label, n.label);
                } else {
                    assert_eq!(x.label, n.label);
                }
            }
        }
    }

    #[test]
    pub fn test_get_started_eliminated_2nodes_stable() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let sm = input.get_train_sms()[0];
        let started_nodes = get_started_eliminated_2nodes(&input.get_annotator(), sm);

        let gold_file = format!("resources/assembling/learning/elimination/{}.started_nodes.json", sm.id);
//        serialize_json(&started_nodes, &gold_file);
        let gold_started_nodes: Vec<(Graph, Vec<String>)> = deserialize_json(&gold_file);

        assert_eq!(started_nodes, gold_started_nodes);
    }
}