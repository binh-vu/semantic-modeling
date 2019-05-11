use assembling::models::mrr::MRRModel;
use rdb2rdf::models::semantic_model::*;
use algorithm::prelude::*;
use rayon::prelude::*;
use settings::{ Settings, conf_search, conf_predicting };
use assembling::searching::*;
use im::OrdSet as IOrdSet;
use std::cmp::Ordering;
use assembling::searching::beam_search::*;
use assembling::searching::discovery::*;
use assembling::ranking::{Ranking, MicroRanking};
use assembling::learning::elimination::cascade_remove;
use fnv::FnvHashSet;
use settings::conf_search::DiscoverMethod;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use std::collections::HashSet;
use assembling::auto_label;
use evaluation_metrics::semantic_modeling::Bijection;
use assembling::auto_label::alignment::align_graph;
use evaluation_metrics::semantic_modeling::DataNodeMode;
use assembling::searching::banks::data_structure::int_graph::INodeData;
use assembling::searching::banks::data_structure::int_graph::TAG_FROM_NEW_SOURCE;
use assembling::searching::banks::data_structure::int_graph::IEdgeData;
use assembling::searching::discovery::constraint_space::Bijection as IBijection;
#[derive(Clone)]
pub struct SimulatedUser {
    n_corrections: i32,
    n_selections: i32,
    n_selections_as_corrections: i32,

    history: Vec<String>
}

impl SimulatedUser {
    pub fn new() -> SimulatedUser {
        SimulatedUser {
            n_corrections: 0,
            n_selections: 0,
            n_selections_as_corrections: 0,
            history: Vec::new(),
        }
    }

    pub fn change_link(&mut self) {
        self.n_corrections += 1;
    }

    pub fn remove_link(&mut self) {
        self.n_corrections += 1;
    }

    pub fn select_correct_model(&mut self, prev_model: &Graph, new_model: &Graph) {
        self.n_selections += 1;
    }

    pub fn get_n_corrections(&self) -> i32 {
        return self.n_corrections;
    }

    pub fn get_n_selections(&self) -> i32 {
        return self.n_selections;
    }

    pub fn get_n_selections_as_corrections(&self) -> i32 {
        return self.n_selections_as_corrections;
    }
}

pub fn simulated_interactive_modeling<'a>(model: &MRRModel<'a>, evaluate_sms: &[&SemanticModel]) -> Vec<(String, SimulatedUser)> {
    let settings = Settings::get_instance();
    let conf_predicting::SearchMethod::BeamSearch(ref search_settings) = settings.predicting.search_method;

    let int_graph = IntGraph::new(&model.annotator.train_sms);

    let conf = if let DiscoverMethod::ConstraintSpace(ref conf) = search_settings.discovery {
        conf
    } else {
        panic!("invalid discovery method")
    };

    let results = evaluate_sms
        .iter().skip(6)
        .map(|sm| {
            let prob_sms = |graphs: Vec<Graph>, bijections: &[constraint_space::Bijection], args: &IntTreeSearchArgs| model.predict_sm_probs(&sm.id, graphs);
            let adapted_int_graph = int_graph.adapt_new_source(sm, None);
            let compare_func: CompareSearchNodeFunc<IntTreeSearchNodeData> = &default_compare_search_node;
            let mut args = SearchArgs {
                sm_id: &sm.id,
                beam_width: search_settings.beam_width,
                n_results: search_settings.n_results,
                enable_tracking: true,
                tracker: Vec::new(),
                discover: &constraint_space::discover,
                should_stop: &no_stop,
                extra_args: IntTreeSearchArgs::default_with_prob_candidate_sms(sm, conf, &prob_sms, adapted_int_graph),
                compare_search_node: compare_func
            };

            let started_nodes = constraint_space::get_started_pk_nodes(sm, &model.annotator, &mut args.extra_args);
            let results = simulate_interacting_search(started_nodes, &mut args, sm);

            (sm.id.clone(), results)
        })
        .collect::<Vec<_>>();

    return results;
}

// from now is the code for simulating interactive modeling..
pub fn simulate_interacting_search(starts: Vec<SearchNode<IntTreeSearchNodeData>>, args: &mut SearchArgs<IntTreeSearchNodeData, IntTreeSearchArgs>, gold_sm: &SemanticModel) -> SimulatedUser {
    let mut user = SimulatedUser::new();
    let mut top_k = 3;
    let mut iter_no = 0;

    // ##############################################
    // Add very first nodes to kick off BEAM SEARCH
    let mut current_exploring_nodes: ExploringNodes<IntTreeSearchNodeData> = Default::default();
    let mut next_exploring_nodes: ExploringNodes<IntTreeSearchNodeData> = starts;
    
    // === [DEBUG] DEBUG CODE START HERE ===
    println!("[DEBUG] at interactive_modeling.rs");
    graph2pdf(&gold_sm.graph, "/tmp/sm_debugging/model.gold.pdf", None);
    graph2pdf(&args.extra_args.int_graph.graph, "/tmp/sm_debugging/model.int_graph.pdf", None);
    // === [DEBUG] DEBUG CODE END   HERE ===
    
    // ##############################################
    // Start BEAM SEARCH!!
    loop {
        iter_no += 1;
        trace!("=== ({}) BEAMSEARCH ===> Doing round: {}", args.sm_id, iter_no);

        if iter_no > 1 {
            // first round, use start nodes
            next_exploring_nodes = (*args.discover)(&current_exploring_nodes, &mut args.extra_args);
        };
        // sort next nodes by their score, higher is better
        next_exploring_nodes.sort_by(args.compare_search_node);
        next_exploring_nodes.truncate(top_k);

        if next_exploring_nodes.len() == 0 {
            // we don't have other nodes from here, so we have to break, and we fill the remaining attributes
            // with information from gold semantic models
            break;
        }

        // now comparing with the ground-truth to detect if it is correct
        let curated_next_state;
        let curated_state = if iter_no > 1 {
            current_exploring_nodes.remove(0)
        } else {
            IntTreeSearchNode::new("".to_owned(), 1.0, IntTreeSearchNodeData::new(
                Graph::new_like(&gold_sm.graph),
                Default::default(),
                Default::default()
            ))
        };

        let pred_sm = next_exploring_nodes[0].get_graph();
        let mrr_label = auto_label::label(&gold_sm.graph, &pred_sm, 50000).expect("Need to increase max-permutation");
        if mrr_label.precision == 1.0 {
            trace!("\t\t user don't need to correct anything");
            curated_next_state = next_exploring_nodes[0].clone();
        } else {
            let best_node = next_exploring_nodes.iter().max_by_key(|n| {
                let mrr_lbl = auto_label::label(&gold_sm.graph, n.get_graph(), 50000).expect("Need to increase max-permutation");
                let score: u32 = mrr_lbl.edge2label.iter().map(|lbl| *lbl as u32).sum();
                return score;
            }).unwrap();
            let best_mrr_lbl = auto_label::label(&gold_sm.graph, best_node.get_graph(), 50000).expect("Need to increase max-permutation");

            if best_mrr_lbl.precision == 1.0 {
                trace!("\t\t user select correct model from top 3 suggestions");
                curated_next_state = best_node.clone();
                user.select_correct_model(curated_state.get_graph(), best_node.get_graph());
            } else {
                // === [DEBUG] DEBUG CODE START HERE ===
                println!("[DEBUG] at predicting.rs");
                graph2pdf(best_node.get_graph(), &format!("/tmp/sm_debugging/model.step_{}.recommendation.pdf", iter_no), None);
                
                // === [DEBUG] DEBUG CODE  END  HERE ===

                // invoke user to fix this task
                let new_curated_graph = simulate_user_actions(curated_state.get_graph(), best_node.get_graph(), &gold_sm.graph, &mut user);
                // === [DEBUG] DEBUG CODE START HERE ===
                println!("[DEBUG] at interactive_modeling.rs");
                // === [DEBUG] DEBUG CODE END   HERE ===

                curated_next_state = realign_curated_graph_to_int_graph(
                    &curated_state, best_node.clone(),
                    new_curated_graph, &mut args.extra_args);
            }
        }

        // === [DEBUG] DEBUG CODE START HERE ===
        println!("[DEBUG] at interactive_modeling.rs");
        graph2pdf(curated_next_state.get_graph(), &format!("/tmp/sm_debugging/model.step_{}.corrected.pdf", iter_no), None);
        // === [DEBUG] DEBUG CODE END   HERE ===
        
        current_exploring_nodes = vec![curated_next_state];
    }

    // check if all link have been matched otherwise we just add up a new links
    let pred_sm = current_exploring_nodes.remove(0);
    assert_eq!(pred_sm.get_graph().n_nodes, gold_sm.graph.n_nodes);

    user
}

/// Simulate a user to update current suggested graph
pub fn simulate_user_actions(curated_sm: &Graph, pred_sm: &Graph, gold_sm: &Graph, user: &mut SimulatedUser) -> Graph {
    let pred2curated = auto_label::label_no_ambiguous(pred_sm, curated_sm, 50000).unwrap();
    let mut new_g = curated_sm.clone();

    for attr in pred_sm.iter_data_nodes() {
        if curated_sm.iter_nodes_by_label(&attr.label).next().is_none() {
            // this is a new attribute, user start checking from here
            let e_prime = attr.first_incoming_edge(pred_sm).unwrap();
            let new_attr_id = new_g.add_node(Node::new(attr.kind, attr.label.clone()));

            new_g = simulate_user_action_on_one_edge(
                e_prime, new_g.get_node_by_id(new_attr_id),
                gold_sm.iter_nodes_by_label(&attr.label).next().unwrap(),
                &new_g, pred_sm, gold_sm, &pred2curated.bijection, user
            );
            break;
        }
    }

    // // === [DEBUG] DEBUG CODE START HERE ===
    println!("[DEBUG] at interactive_modeling.rs");
    graph2pdf(&new_g, "/tmp/sm_debugging/model.fixing_step1.pdf", None);
    // // === [DEBUG] DEBUG CODE END   HERE ===

    // check if merge need to merge from the root
    // one way to check is new_g have two root now
    let new2old_curated = auto_label::label_no_ambiguous(&new_g, curated_sm, 50000).unwrap();
    let gold2new_curated = auto_label::label_no_ambiguous(gold_sm, &new_g, 50000).unwrap();
    debug_assert_eq!(new2old_curated.precision, 1.0);
    debug_assert_eq!(gold2new_curated.precision, 1.0);
    let root_prime_id = new_g.get_first_root_node().unwrap().id;
    let prev_root_prime = curated_sm.get_first_root_node().unwrap();

    if new_g.iter_nodes().filter(|n| n.n_incoming_edges == 0).count() > 1 {
        // okay user also need to start fixing from here
        let e_prime = pred_sm
            .get_node_by_id(pred2curated.bijection.to_x(prev_root_prime.id) as usize)
            .first_incoming_edge(pred_sm).unwrap();
        let prev_root = gold_sm.get_node_by_id(gold2new_curated.bijection.to_x(
            new2old_curated.bijection.to_x(prev_root_prime.id) as usize) as usize
        );
        new_g = simulate_user_action_on_one_edge(
            e_prime, prev_root_prime, prev_root, &new_g,
            pred_sm, gold_sm, &pred2curated.bijection, user
        );

        // now they may still need to add more link because the first just update links
        
        
        // === [DEBUG] DEBUG CODE START HERE ===
        println!("[DEBUG] at interactive_modeling.rs");
        println!("[DEBUG] new_g.n_edges = {} fixing second part", new_g.n_edges);
        // === [DEBUG] DEBUG CODE END   HERE ===
    }

    // // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] at interactive_modeling.rs");
    graph2pdf(&new_g, "/tmp/sm_debugging/model.fixing_step2.pdf", None);
    // // === [DEBUG] DEBUG CODE END   HERE ===
    
    // now verify the correctness of this models
    debug_assert!(is_perfect_match(gold_sm, &new_g));
    debug_assert_eq!(new_g.iter_nodes().filter(|n| n.n_incoming_edges == 0).count(), 1);
    return new_g;
}

/// Simulate a user action to fix one edges
pub fn simulate_user_action_on_one_edge(e_prime: &Edge, v_prime: &Node, v: &Node, curated_sm: &Graph, pred_sm: &Graph, gold_sm: &Graph, pred2curated: &Bijection, user: &mut SimulatedUser) -> Graph {
    let e = v.first_incoming_edge(gold_sm).unwrap();
    let u = e.get_source_node(gold_sm);

    let u_prime = e_prime.get_source_node(pred_sm);

    let mut take_action = false;
    let mut action_node = None;

    // user fix source node before fixing links
    if u.label == u_prime.label {
        // the label is matched, check if u_prime map to existing node in curated_sm
        let mut exist_u_pp = false;
        for u_pp in curated_sm.iter_nodes_by_label(&u_prime.label) {
            let hypothesis_sm = add_link_to_graph(curated_sm, u_pp.id, &e.label, v_prime.id);
            if is_perfect_match(gold_sm, &hypothesis_sm) {
                if u_pp.id != u_prime.id {
                    // user need to change u_prime to u_prime prime, otherwise don't need to do anything
                    take_action = true;
                    action_node = Some((u_prime.id, u_pp.id));
                }
                exist_u_pp = true;
                break;
            }
        }

        if !exist_u_pp {
            // it doesn't map to an existing node in curated sm so u_prime must be mapped to new node
            // we check if it has not, we have to fix it
            // pred_sm as gold model, curated_sm as predicted model therefore, x is nodes in pred_sm, x_prime is nodes in
            // curated_sm. so we have to check if u_prime can map to any node x_prime in curated_sm.
            if pred2curated.has_x_prime(u_prime.id) {
                take_action = true;
                // map u_prime to new node
                action_node = Some((u_prime.id, gold_sm.n_nodes + 1));
            }
        }
    } else {
        // the label is not match,
        take_action = true;

        // repeat the procedure, find if u map to existing node in curated sm
        for u_pp in curated_sm.iter_nodes_by_label(&u.label) {
            let hypothesis_sm = add_link_to_graph(curated_sm, u_pp.id, &e.label, v_prime.id);
            if is_perfect_match(gold_sm, &hypothesis_sm) {
                // u_prime map to u_pp
                action_node = Some((u_prime.id, u_pp.id));
                break;
            }
        }

        if action_node.is_none() {
            // it doesn't map to an existing node in curated_sm, so it has to map to new node
            // we need to check if it is already in curated sm otherwise we need to create new node
            if pred2curated.has_x_prime(u_prime.id) {
                action_node = Some((u_prime.id, gold_sm.n_nodes + 1));
            }
        }
    }

    // check link label match
    if e.label != e_prime.label {
        take_action = true;
    }

    if take_action {
        // user need to change link
        user.change_link();
        let mut new_g = curated_sm.clone();
        if let Some((old_id, new_id)) = action_node {
            let new_u_id = if new_id == gold_sm.n_nodes + 1 {
                // need to map from old link to new link
                new_g.add_node(Node::new(u.kind, u.label.clone()))
            } else {
                new_id
            };

            new_g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), new_u_id, v_prime.id));

            // === [DEBUG] DEBUG CODE START HERE ===
            println!("[DEBUG] at interactive_modeling.rs -- user change old id to new id");
            graph2pdf(&new_g, "/tmp/sm_debugging/tmp.pdf", None);
            // === [DEBUG] DEBUG CODE END   HERE ===
            
            let mut next_e = u_prime.first_incoming_edge(&pred_sm);
            if next_e.is_none() {
                // we stop here because we cannot go further
                new_g
            } else if new_id != gold_sm.n_nodes + 1 {
                // or we map to an existing node
                // so that we need to remove all link
                while next_e.is_some() {
                    user.remove_link();
                    next_e = next_e.unwrap().get_source_node(&pred_sm).first_incoming_edge(&pred_sm);
                }
                new_g
            } else {
                simulate_user_action_on_one_edge(
                    next_e.unwrap(), new_g.get_node_by_id(new_u_id),
                    u, &new_g, pred_sm, gold_sm, pred2curated, user
                )
            }
        } else {
            // action node is none, which mean the label of link is incorrect, so user modify a incorrect label link
            // === [DEBUG] DEBUG CODE START HERE ===
            println!("[DEBUG] at interactive_modeling.rs -- user need to add new link but can add new graph");
            graph2pdf(&new_g, "/tmp/sm_debugging/tmp.pdf", None);
            println!("[DEBUG] u_prime.id = {}", u_prime.id);
            println!("[DEBUG] pred2curated = {:?}", pred2curated);
            // === [DEBUG] DEBUG CODE END   HERE ===

            let new_u_id = if pred2curated.has_x_prime(u_prime.id) {
                pred2curated.to_x_prime(u_prime.id) as usize
            } else {
                new_g.add_node(Node::new(u.kind, u.label.clone()))
            };

            println!("[DEBUG] new_u_id = {}", new_u_id);
            new_g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), new_u_id, v_prime.id));

            let mut next_e = u_prime.first_incoming_edge(&pred_sm);
            if next_e.is_none() {
                // we stop here because we cannot go further
                new_g
            } else if pred2curated.has_x_prime(u_prime.id) {
                // we map to an existing node
                // so that we need to remove all parent links
                while next_e.is_some() {
                    user.remove_link();
                    next_e = next_e.unwrap().get_source_node(&pred_sm).first_incoming_edge(&pred_sm);
                }
                new_g
            } else {
                simulate_user_action_on_one_edge(
                    next_e.unwrap(), new_g.get_node_by_id(new_u_id),
                    u, &new_g, pred_sm, gold_sm, pred2curated, user
                )
            }
        }
    } else {
        // user don't need to do anything, we add this edge to the curated graph
        let mut new_g = curated_sm.clone();
        let new_u_id = new_g.add_node(Node::new(u.kind, u.label.clone()));
        new_g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(),
            new_u_id, v_prime.id
        ));

        let next_e = u_prime.first_incoming_edge(&pred_sm);
        if next_e.is_none() {
            // then we stop here because we cannot go further
            new_g
        } else {
            simulate_user_action_on_one_edge(
                next_e.unwrap(), new_g.get_node_by_id(new_u_id),
                u, &new_g, pred_sm, gold_sm, pred2curated, user
            )
        }
    }
}

fn add_link_to_graph(curated_sm: &Graph, u_pp_id: usize, e_lbl: &str, v_pp_id: usize) -> Graph {
    let mut new_g = curated_sm.clone();
    new_g.add_edge(Edge::new(EdgeType::Unspecified, e_lbl.to_owned(), u_pp_id, v_pp_id));
    return new_g
}

fn is_perfect_match(gold_sm: &Graph, pred_sm: &Graph) -> bool {
    let result = auto_label::label_no_ambiguous(gold_sm, pred_sm, 50000);
    // // === [DEBUG] DEBUG CODE START HERE ===
    // println!("[DEBUG] at interactive_modeling.rs");
    // graph2pdf(gold_sm, "/tmp/sm_debugging/a.pdf", None);
    // graph2pdf(pred_sm, "/tmp/sm_debugging/b.pdf", None);
    // // === [DEBUG] DEBUG CODE END   HERE ===
    
    result.expect("There is no more than one answer to label the graph").precision == 1.0
}

/// Realign a newly curated graph to int graph
fn realign_curated_graph_to_int_graph(curated_state: &SearchNode<IntTreeSearchNodeData>, mut next_incorrect_sm: SearchNode<IntTreeSearchNodeData>, next_curated_sm: Graph, extra_args: &mut IntTreeSearchArgs) -> SearchNode<IntTreeSearchNodeData> {
    // there are part of the graph already have bijections, and the other part of the graph doesn't have it
    // we need to separate them so we can align much faster
    // now haven't do it yet

    let mut bijections: Vec<IBijection> = Vec::new();
    let int_graph_norms = extra_args.int_graph.to_normal_graph();
    {
        let mut alignments = align_graph(&int_graph_norms, &next_curated_sm, DataNodeMode::IgnoreLabelDataNode, 100000);
        assert!(alignments.len() > 0);
        let mut update_index = false;

        // if there are node that doesn't appear in the integration graph, we need to add it back
        // notice because of this, because of multiple bijections, the graph may expand quickly
        // we can do better!
        for mut alignment in alignments {
            println!("[DEBUG] loop through alignment: {:?}\n\tprecision={:?}", alignment.3, alignment.1);

            for n_prime in next_curated_sm.iter_nodes() {
                if !alignment.3.has_x(n_prime.id) {
                    if let Some(e_prime) = n_prime.first_incoming_edge(&next_curated_sm) {
                        // we need to propagate, a quick solution is assuming we don't need to
                        assert!(alignment.3.has_x(e_prime.source_id));
                        let pn_id = extra_args.int_graph.graph.get_node_by_id(alignment.3.to_x(e_prime.source_id) as usize).id;
                        let new_n_id = extra_args.int_graph.graph.add_node(Node::new_with_data(
                            n_prime.kind, n_prime.label.clone(), INodeData::new("USER_UPDATED".to_owned())));
                        extra_args.int_graph.graph.add_edge(Edge::new_with_data(
                            e_prime.kind, e_prime.label.clone(), pn_id, new_n_id,
                            IEdgeData::new(TAG_FROM_NEW_SOURCE.to_owned())
                        ));

                        println!("[DEBUG] new_n_id = {}, n_prime.id = {}", new_n_id, n_prime.id);
                        alignment.3.append_x(new_n_id, n_prime.id);
                        update_index = true;
                    } else {
                        unimplemented!()
                    }
                }
            }

            // build prime2x because it is ignored by data node ignore label mode
            let mut prime2x: Vec<i32> = alignment.3.prime2x.clone();
            let parent_x_primes = (0..prime2x.len())
                .filter(|&x_prime| prime2x[x_prime] == -999999)
                .map(|x_prime| {
                    next_curated_sm.get_node_by_id(x_prime).first_incoming_edge(&next_curated_sm).unwrap().source_id
                })
                .collect::<HashSet<usize>>();

            for px_prime_id in parent_x_primes {
                let px_prime = next_curated_sm.get_node_by_id(px_prime_id);
                assert!(px_prime.is_class_node());
                let px_id = prime2x[px_prime_id];
                println!("[DEBUG] px_id = {}", px_id);
                println!("[DEBUG] px_prime_id = {}", px_prime_id);
                let px = extra_args.int_graph.graph.get_node_by_id(px_id as usize);
                let mut used_x = HashSet::new();
                for e_prime in px_prime.iter_outgoing_edges(&next_curated_sm) {
                    if !e_prime.get_target_node(&next_curated_sm).is_data_node() {
                        continue;
                    }

                    for e in px.iter_outgoing_edges(&extra_args.int_graph.graph) {
                        let x = e.get_target_node(&extra_args.int_graph.graph);
                        println!("[DEBUG] e.label = {}, e_prime.label = {}", e.label, e_prime.label);
                        if x.is_data_node() && e_prime.label == e.label && !used_x.contains(&e.target_id) {
                            prime2x[e_prime.target_id] = e.target_id as i32;
                            used_x.insert(e.target_id);
                        }
                    }
                }
            }

            println!("[DEBUG] prime2x = {:?}", prime2x);
            for &x in &prime2x {
                println!("x = {}", x);
                debug_assert!(x >= 0);
            }

            let bijection = IBijection::from_pairs(prime2x.into_iter().map(|x| x as usize).collect::<Vec<_>>());
            bijections.push(bijection);
        }

        if update_index {
            extra_args.update_index();
        }
    }

    next_incorrect_sm.data.update_graph(next_curated_sm);
    next_incorrect_sm.data.update_bijections(bijections);
    next_incorrect_sm.id = next_incorrect_sm.data.get_id();
    next_incorrect_sm
}