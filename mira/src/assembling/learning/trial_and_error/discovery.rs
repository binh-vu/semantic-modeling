use utils::dict_has;
use utils::dict_get;
use std::sync::Arc;
use assembling::learning::trial_and_error::data_structure::Mount;
use std::collections::HashSet;
use algorithm::prelude::*;
use assembling::models::annotator::Annotator;
use assembling::learning::trial_and_error::data_structure::TrialErrorNodeData;
use assembling::learning::trial_and_error::data_structure::TrialErrorSearchArgs;
use assembling::learning::trial_and_error::data_structure::TrialErrorSearchNode;
use im::HashSet as IHashSet;
use im::OrdSet as IOrdSet;
use rdb2rdf::models::semantic_model::SemanticModel;
use std::iter::FromIterator;
use std::ops::Deref;
use assembling::searching::beam_search::*;
use itertools::Itertools;
use std::rc::Rc;
use std::slice::Iter;


// /// Started nodes is a set of primary keys
// pub fn get_started_nodes<'a>(annotator: &'a Annotator, sm: &SemanticModel, args: &TrialErrorSearchArgs<'a>) -> Vec<TrialErrorSearchNode> {
//     let mut search_seeds = vec![];
//     let attributes = IHashSet::from_iter(sm.attrs.iter().map(|a| a.label.clone()));

//     // create set of seed nodes from primary keys only
//     for attr in &sm.attrs {
//         for stype in &attr.semantic_types {
//             if annotator.primary_key.get_primary_key(&stype.class_uri) == stype.predicate {
//                 // this is primary key, we start from here
//                 let mut g = Graph::new(sm.id.clone(), true, true, false);
//                 let source_id = g.add_node(Node::new(NodeType::ClassNode, stype.class_uri.clone()));
//                 let target_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));
//                 g.add_edge(Edge::new(EdgeType::Unspecified, stype.predicate.clone(), source_id, target_id));
//                 let mut seed = TrialErrorNodeData::new(
//                     attributes.remove(&attr.label),
//                     Some((source_id, 0)),
//                     g);

//                 if !seed.has_mount_candidates(seed.mount.as_ref().unwrap(), args) {
//                     // advance the mount if we don't have any other candidates of this mount
//                     seed.mount = seed.advance_to_non_empty_mount(args);
//                 }

//                 // score is the semantic type score, we can do something more sophisticated like statistic
//                 search_seeds.push(seed.into_search_node(stype.score as f64));
//             }
//         }
//     }

//     search_seeds
// }


/// Started nodes is a set of primary keys
pub fn get_started_nodes<'a>(annotator: &'a Annotator, sm: &SemanticModel, args: &TrialErrorSearchArgs<'a>) -> Vec<TrialErrorSearchNode> {
    let mut search_seeds = vec![];
    let attributes = IOrdSet::from_iter(sm.attrs.iter().map(|a| a.label.clone()));
    let mut class_uris: HashSet<String> = Default::default();

    // create set of seed nodes from primary keys only
    for attr in &sm.attrs {
        for stype in &attr.semantic_types {
            class_uris.insert(stype.class_uri.clone());
        }
    }

    // let mut ordered_class_uris = class_uris.iter().cloned().collect::<Vec<_>>();
    // ordered_class_uris.sort_unstable();
    // println!("[DEBUG] ordered_class_uris = {:?}", ordered_class_uris);
    // for class_uri in ordered_class_uris.iter() {
    for class_uri in class_uris.iter() {
        // this is primary key, we start from here
        let mut g = Graph::new(sm.id.clone(), true, true, false);
        let source_id = g.add_node(Node::new(NodeType::ClassNode, class_uri.clone()));
        let mut seed = TrialErrorNodeData::new(
            attributes.clone(),
            Some(Mount::new(source_id, 0, false)),
            g);

        // score is the semantic type score, we can do something more sophisticated like statistic
        search_seeds.push(seed.into_search_node(annotator.statistic.p_n(class_uri, 0.0) as f64));
    }

    search_seeds
}


pub fn get_all_matches<'a: 'a1, 'a1>(mnt_subj: &Node, mnt_pred: &str, attrs: &'a1 IOrdSet<String>, args: &TrialErrorSearchArgs<'a>) -> Vec<&'a1 String> {
    let mut match_attrs = attrs.iter()
        .filter(|attr_lbl| {
            // println!("[DEBUG] attr_lbl = {}, mnt_subj: {}, mnt_pred: {}", attr_lbl, mnt_subj.label, mnt_pred);
            for stype in args.get_attr_stypes(&attr_lbl).iter() {
                if stype.class_uri == mnt_subj.label && &stype.predicate == mnt_pred {
                    return true;
                }
            }
            return false;
        })
        .collect::<Vec<_>>();

    // so we try to mount all of them into new mnt
    if match_attrs.len() > args.max_n_duplications {
        let mut match_attrs_w_score = match_attrs.into_iter()
            .map(|attr_lbl| {
                for stype in args.get_attr_stypes(&attr_lbl).iter() {
                    if stype.class_uri == mnt_subj.label && &stype.predicate == mnt_pred {
                        return (attr_lbl, stype.score);
                    }
                }

                return (attr_lbl, 0.0);
            })
            .collect::<Vec<_>>();

        match_attrs_w_score.sort_by(|(_, w1), (_, w2)| w2.partial_cmp(w1).unwrap());
        match_attrs_w_score.truncate(args.max_n_duplications);

        match_attrs = match_attrs_w_score.into_iter().map(|(lbl, w)| lbl).collect::<Vec<_>>();
    }

    match_attrs
}


pub fn is_class_mount(mnt_subj: &Node, mnt_pred: &String, args: &TrialErrorSearchArgs) -> bool {
    return dict_has(&args.trial_error_exploring.o_given_sl, &mnt_subj.label, mnt_pred);
}

pub fn iter_mount_classes<'a>(mnt_subj: &Node, mnt_pred: &String, args: &'a TrialErrorSearchArgs) -> &'a [String] {
    &dict_get(&args.trial_error_exploring.o_given_sl_ordered, &mnt_subj.label, mnt_pred).unwrap()
}

pub fn generate_next_states<'a>(mut mount: Mount, attrs: &IOrdSet<String>, current_graph: &Graph, args: &TrialErrorSearchArgs<'a>) -> (Vec<(IOrdSet<String>, Mount)>, Vec<Graph>) {
    let mut next_states_data = Vec::new();
    let mut next_states_graphs = Vec::new();

    if mount.is_done {
        loop {
            match mount.next_mount(current_graph, args) {
                None => {
                    return (next_states_data, next_states_graphs);
                },
                Some(mnt) => {
                    mount = mnt;
                    if !mount.is_done {
                        break;
                    }
                }
            }
        }
    }

    let (mnt_subj, mnt_pred) = mount.unroll(&current_graph, args);
    // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] mnt_subj.id = {}, mnt_subj.label = {}, mnt_pred = {}", mnt_subj.id, mnt_subj.label, mnt_pred);
    // === [DEBUG] DEBUG CODE END   HERE ===
    
    if is_class_mount(mnt_subj, mnt_pred, args) {
        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] at discovery.rs");
        // println!("[DEBUG] mnt_subj.label = {}, mnt_pred = {}", mnt_subj.label, mnt_pred);
        // println!("[DEBUG] mount_classes = {:?}", iter_mount_classes(mnt_subj, mnt_pred, args).collect::<Vec<_>>());
        // === [DEBUG] DEBUG CODE  END  HERE ===
        // println!("[DEBUG] iter_mount_classes(mnt_subj, mnt_pred, args) = {:?}", iter_mount_classes(mnt_subj, mnt_pred, args));
        
        for class_uri in iter_mount_classes(mnt_subj, mnt_pred, args) {
            let mut new_g = current_graph.clone();
            let target_id = new_g.add_node(Node::new(NodeType::ClassNode, class_uri.clone()));
            new_g.add_edge(Edge::new(EdgeType::Unspecified, mnt_pred.clone(), mnt_subj.id, target_id));
            let mut new_mount = Mount::new(target_id, 0, false);

            if new_g.get_node_by_id(mnt_subj.id).iter_outgoing_edges(&new_g).all(|e| e.get_target_node(&new_g).is_class_node()) {
                // going to ignore this because it create infinited recursive
                continue;
            }

            let (mut sub_next_states_data, mut sub_next_states_graphs) = generate_next_states(new_mount, attrs, &new_g, args);
            
            // have to filter out the one has middle node which doesn't have any data node
            // because it create ambiguous example and infinited recursive
            // name -- Club -- Player -- Club -- inLeague -- League -- name
            for (i, sub_g) in sub_next_states_graphs.into_iter().enumerate().rev() {
                let has_no_data_node = sub_g.iter_class_nodes()
                    .any(|n| {
                        n.iter_outgoing_edges(&sub_g)
                            .all(|e| e.get_target_node(&sub_g).is_class_node())
                    });

                if !has_no_data_node {
                    let mut elem = sub_next_states_data.swap_remove(i);
                    elem.1 = mount.clone();
                    next_states_data.push(elem);
                    next_states_graphs.push(sub_g);
                }
            }
        }

        if next_states_data.len() == 0 {
            // no more elements, mean this mount is done!
            // === [DEBUG] DEBUG CODE START HERE ===
            // println!("[DEBUG] move on to next mount");
            // === [DEBUG] DEBUG CODE END   HERE ===
            
            match mount.next_mount(current_graph, args) {
                None => {
                    return (next_states_data, next_states_graphs);
                },
                Some(mnt) => {
                    return generate_next_states(mnt, attrs, current_graph, args);
                }
            }
        }
    } else {
        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] attrs = {:?}", attrs);
        // === [DEBUG] DEBUG CODE END   HERE ===
        let match_attrs = get_all_matches(mnt_subj, mnt_pred, &attrs, args);
        // next states would be all possible graph can generate from this mount
        // for example, the mount_subj is Player, and the mount_pred is age.
        // we have two attributes x and y can fit to age. so we generate 3 graphs:
        // Player - age - x, Player - age - y, Player - age - x - age - y.
        // the total is: C[1][n] + C[2][n] + ... + C[n][n]

        if match_attrs.len() == 0 {
            // === [DEBUG] DEBUG CODE START HERE ===
            // println!("[DEBUG] move on to next mount");
            // === [DEBUG] DEBUG CODE END   HERE ===
            match mount.next_mount(current_graph, args) {
                None => {
                    return (next_states_data, next_states_graphs);
                },
                Some(mnt) => {
                    return generate_next_states(mnt, attrs, current_graph, args);
                }
            }
        }

        for i in 1..(match_attrs.len() + 1) {
            match_attrs.iter().combinations(i)
                .foreach(|attr_lbls| {
                    // println!("[DEBUG] attr_lbls = {:?}", attr_lbls);
                    let mut g = current_graph.clone();
                    let mut remained_attributes = attrs.without(attr_lbls[0].as_str());

                    for attr_lbl in &attr_lbls {
                        let target_id = g.add_node(Node::new(NodeType::DataNode, attr_lbl.as_str().to_owned()));
                        g.add_edge(Edge::new(EdgeType::Unspecified, mnt_pred.to_owned(), mnt_subj.id, target_id));
                        remained_attributes.remove(attr_lbl.as_str());
                    }

                    let mut new_mount = mount.clone();
                    new_mount.is_done = true;
                    next_states_data.push((remained_attributes, new_mount));
                    next_states_graphs.push(g);
                });
        }
    }
    
    (next_states_data, next_states_graphs)
}


pub fn discover<'a>(search_nodes: &ExploringNodes<TrialErrorNodeData>, args: &mut TrialErrorSearchArgs<'a>) -> Vec<TrialErrorSearchNode> {
    let mut next_states_data = Vec::with_capacity(search_nodes.len());
    let mut next_states_graphs = Vec::with_capacity(search_nodes.len());
    let mut next_states = Vec::with_capacity(search_nodes.len());

    for search_node in search_nodes.iter() {
        let mut mount = search_node.data.mount.clone().expect("No mount has been filtered before calling discover function");
        let (mut sub_next_states_data, mut sub_next_states_graphs) = generate_next_states(mount, &search_node.data.remained_attributes, &search_node.data.current_graph, args);
        // === [DEBUG] DEBUG CODE START HERE ===
        // println!("[DEBUG] sub_next_states_data.len() = {}", sub_next_states_data.len());
        // println!("[DEBUG] sub_next_states_graphs.len() = {}", sub_next_states_graphs.len());
        // === [DEBUG] DEBUG CODE END   HERE ===
        next_states_data.append(&mut sub_next_states_data);
        next_states_graphs.append(&mut sub_next_states_graphs);
    }

    // compute the prob
    for ((g, w), (ra, mnt)) in (*args.prob_candidate_sms)(next_states_graphs).into_iter().zip(next_states_data) {
        let mut trial_data = TrialErrorNodeData::new(ra, Some(mnt), g);
        next_states.push(trial_data.into_search_node(w));
    }

    next_states.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    next_states.truncate(args.beam_width);

    // println!("[DEBUG] search_nodes.len() = {}, next_states.len() = {}", search_nodes.len(), next_states.len());

    next_states
}