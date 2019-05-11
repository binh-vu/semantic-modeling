use assembling::models::annotator::Annotator;
use im::hashset::HashSet as IHashSet;
use serde::Serialize;
use algorithm::data_structure::graph::*;
use std::rc::Rc;
use std::collections::*;
use rdb2rdf::models::semantic_model::SemanticModel;
use rdb2rdf::models::semantic_model::Attribute;
use std::ops::Deref;
use serde_json;
use std::fs::File;
use super::*;
use assembling::searching::beam_search::*;


fn select_top_attributes(sm: &SemanticModel, top_n: usize) -> Vec<String> {
    let mut top_attrs = sm.attrs.iter().map(|attr| (&attr.label, attr.semantic_types[0].score)).collect::<Vec<_>>();
    top_attrs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    top_attrs.truncate(top_n);
    top_attrs.into_iter().map(|(lbl, _score)| lbl.clone()).collect()
}

pub fn get_started_nodes(sm: &SemanticModel, top_n: usize, args: &mut GeneralDiscoveryArgs) -> Vec<GeneralDiscoveryNode> {
    let mut top_attrs = select_top_attributes(sm, top_n);
    
    // create graph & graph explorer for each attributes
    let singularity = GraphDiscovery::init(&sm.attrs, args);
    
    // final all possible merged points between every terminal pairs & release it as terminal nodes
    // TOO EXPENSIVE so we just do with top_attrs
    let started_nodes = top_attrs.iter()
        .map(|attr| {
            let idx = singularity.terminals_to_index[attr];
            let g = singularity.g_terminals[idx].clone();

            singularity.merge_terminal(
                attr,
                singularity.g_explorers[idx].clone(),
                g,
                1.0,
            )
        })
        .collect::<Vec<_>>();

    let mut next_graphs = Vec::new();
    let mut next_graph_ids: HashSet<String> = Default::default();
    let mut next_provenance = Vec::new();

    for (node_idx, started_node) in started_nodes.iter().enumerate() {
        for t_j in started_node.remained_terminals.iter() {
            // doing filter to speed up, will remove all merge graph that have more than 3 nodes (because the good result
            // is usually two data nodes connect to one single class nodes
            started_node.make_merge_plans4case1(&t_j)
                .into_iter()
                .filter(|ref mg| mg.get_n_nodes() == 3)
                .for_each(|mg| {
                    let g = mg.proceed_merging();
                    let g_id = get_acyclic_consistent_unique_hashing(&g);
                    if !next_graph_ids.contains(&g_id) {
                        next_graph_ids.insert(g_id);
                        next_graphs.push(mg.proceed_merging());
                        next_provenance.push((node_idx, t_j.clone()));
                    }
                });
        }
    }

    trace!("{} #possible next states", next_graphs.len());
    let next_graph_w_scores = (*args.prob_candidate_sms)(next_graphs);

    // === [DEBUG] DEBG CODE START HERE ===
    // println!("[DEBUG] at general_approach.rs");
    // println!("[DEBUG] started_nodes = {:?}", started_nodes.iter().map(|n| n.current_graph.get_node_by_id(0).label.clone()).collect::<Vec<_>>());
    // println!("[DEBUG] next_graph_scores = {:?}", next_graph_w_scores.iter().map(|s| s.1).collect::<Vec<_>>());
    // use debug_utils::*;
    // draw_graphs(&[next_graph_w_scores[next_graph_w_scores.len() - 1].0.clone()]);
    // === [DEBUG] DEBUG CODE END   HERE ===

    let mut next_states = next_graph_w_scores.into_iter().zip(next_provenance.into_iter()).collect::<Vec<_>>();
    next_states.sort_by(|a, b| (b.0).1.partial_cmp(&(a.0).1).unwrap());
    next_states.truncate(args.beam_width);

    // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] truncated_next_graph_scores = {:?}", next_states.iter().map(|s| (s.0).1).collect::<Vec<_>>());
    // === [DEBUG] DEBUG CODE END   HERE ===

    next_states.into_iter()
        .map(|((g, score), (node_idx, terminal))| {
            let g_explorer = args.graph_explorer_builder.build(&g);
            let graph_discovery = started_nodes[node_idx].merge_terminal(&terminal, g_explorer, g, score);
            SearchNode::new(graph_discovery.get_current_id(), graph_discovery.current_score, graph_discovery)
        })
        .collect()
}


pub fn get_started_pk_nodes<'a>(sm: &SemanticModel, annotator: &Annotator<'a>, args: &mut GeneralDiscoveryArgs) -> Vec<GeneralDiscoveryNode> {
    let mut search_seeds = vec![];

    // create graph & graph explorer for each attributes
    let singularity = GraphDiscovery::init(&sm.attrs, args);

    // create set of seed nodes from primary keys only
    for attr in &sm.attrs {
        for stype in &attr.semantic_types {
            if annotator.primary_key.contains(&stype.class_uri) && annotator.primary_key.get_primary_key(&stype.class_uri) == stype.predicate {
                // this is primary key, we start from here
                let mut g = Graph::new(sm.id.clone(), true, true, false);
                let source_id = g.add_node(Node::new(NodeType::ClassNode, stype.class_uri.clone()));
                let target_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));
                g.add_edge(Edge::new(EdgeType::Unspecified, stype.predicate.clone(), source_id, target_id));

                let g_explorer = args.graph_explorer_builder.build(&g);
                let seed = singularity.merge_terminal(&attr.label, g_explorer, g, stype.score as f64);
                // score is the semantic type score, we can do something more sophisticated like statistic
                search_seeds.push(SearchNode::new(seed.get_current_id(), seed.current_score, seed));
            }
        }
    }

    search_seeds
}

pub fn discover(search_nodes: &ExploringNodes<GraphDiscovery>, args: &mut GeneralDiscoveryArgs) -> Vec<GeneralDiscoveryNode> {
    let mut next_graphs = Vec::new();
    let mut next_graph_ids: HashSet<String> = Default::default();
    let mut next_provenance = Vec::new();

    for (node_idx, search_node) in search_nodes.iter().enumerate() {
        for t_j in search_node.data.remained_terminals.iter() {
            search_node.data.make_merge_plans(&t_j)
                .into_iter()
                .for_each(|mg| {
                    let g = mg.proceed_merging();
                    let g_id = get_acyclic_consistent_unique_hashing(&g);
                    if !next_graph_ids.contains(&g_id) && (*args.pre_sm_filter)(&g) {
                        next_graph_ids.insert(g_id);
                        next_graphs.push(mg.proceed_merging());
                        next_provenance.push((node_idx, t_j.clone()));
                    }
                });
        }
    }

    trace!("{} #possible next states", next_graphs.len());
    let next_graph_w_scores = (*args.prob_candidate_sms)(next_graphs);

    let mut next_states = next_graph_w_scores.into_iter().zip(next_provenance.into_iter()).collect::<Vec<_>>();
    next_states.sort_by(|a, b| (b.0).1.partial_cmp(&(a.0).1).unwrap());
    next_states.truncate(args.beam_width);

    next_states.into_iter()
        .map(|((g, score), (node_idx, terminal))| {
            let g_explorer = args.graph_explorer_builder.build(&g);
            let graph_discovery = search_nodes[node_idx].data.merge_terminal(&terminal, g_explorer, g, score);
            SearchNode::new(graph_discovery.get_current_id(), graph_discovery.current_score, graph_discovery)
        })
        .collect()
}