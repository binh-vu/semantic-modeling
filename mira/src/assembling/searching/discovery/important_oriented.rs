use std::collections::HashSet;
use algorithm::data_structure::graph::graph_util::get_acyclic_consistent_unique_hashing;
use rdb2rdf::models::semantic_model::Attribute;
use assembling::searching::discovery::GraphDiscovery;
use assembling::models::annotator::Annotator;
use assembling::searching::discovery_helper::GraphExplorer;
use algorithm::data_structure::graph::*;

//pub fn get_started_nodes(annotator: &Annotator, attrs: &Vec<Attribute>, args: &mut SearchArgs) -> Vec<SearchNode> {
//    // The started nodes are primary keys of those attributes
//
//    // create graph & graph explorer for each attributes
//    let singularity = GraphDiscovery::init(attrs, args);
//    sm.attrs.iter()
//        .map(|attr| {
//            for stype in &attr.semantic_types {
//                let pk = annotator.primary_key.get_primary_key(stype.class);
//                if pk == stype.predicate {
//                    // add this attribute with the predicate
//                    let idx = singularity.terminals_to_index[attr];
//                    let g = singularity.g_terminals[idx].clone();
//                    let mut g_explorer = GraphExplorer::new(g);
//
//                    let nid = g_explorer.add_node(Node::new(NodeType::ClassNode, stype.class), 0);
//                    // id of attribute always 0
//                    g_explorer.add_edge(Edge::new(EdgeType::Unspecified, stype.predicate, nid, 0));
//
//                    let search_node = singularity.merge_terminal(attr, g_explorer, g, stype.score);
//                    // return search node
//                }
//            }
//        });
//
//    // for each started nodes, base on the class, find out the next
//
//    // final all possible merged points between every terminal pairs & release it as terminal nodes
//    // TOO EXPENSIVE so we just do with top_attrs
//    let started_nodes = top_attrs.iter()
//        .map(|attr| {
//
//
//            singularity.merge_terminal(
//                attr,
//                singularity.g_explorers[idx].clone(),
//                g,
//                1.0
//            )
//        })
//        .collect::<Vec<_>>();
//
//    let mut next_graphs = Vec::new();
//    let mut next_graph_ids: HashSet<String> = Default::default();
//    let mut next_provenance = Vec::new();
//
//
//    for (node_idx, started_node) in started_nodes.iter().enumerate() {
//        for t_j in started_node.remained_terminals.iter() {
//            // doing filter to speed up, will remove all merge graph that have more than 3 nodes (because the good result
//            // is usually two data nodes connect to one single class nodes
//            started_node.make_merge_plans4case1(&t_j)
//                .into_iter()
//                .filter(|ref mg| mg.get_n_nodes() == 3)
//                .for_each(|mg| {
//                    let g = mg.proceed_merging();
//                    let g_id = get_acyclic_consistent_unique_hashing(&g);
//                    if !next_graph_ids.contains(&g_id) {
//                        next_graph_ids.insert(g_id);
//                        next_graphs.push(mg.proceed_merging());
//                        next_provenance.push((node_idx, t_j.clone()));
//                    }
//                });
//        }
//    }
//
//    trace!("{} #possible next states", next_graphs.len());
//    let next_graph_w_scores = (*args.prob_candidate_sms)(next_graphs);
//
//    let mut next_states = next_graph_w_scores.into_iter().zip(next_provenance.into_iter()).collect::<Vec<_>>();
//    next_states.sort_by(|a, b| (b.0).1.partial_cmp(&(a.0).1).unwrap());
//    next_states.truncate(args.beam_width);
//
//    next_states.into_iter()
//        .map(|((g, score), (node_idx, terminal))| {
//            let g_explorer = args.graph_explorer_builder.build(&g);
//            SearchNode::new(started_nodes[node_idx].merge_terminal(&terminal, g_explorer, g, score))
//        })
//        .collect()
//}