use std::collections::HashSet;
pub use self::data_structure::*;
pub use self::early_stopping::*;
pub use self::search_filter::*;

mod data_structure;
mod early_stopping;
mod search_filter;

pub fn beam_search<'a, T, A>(starts: Vec<SearchNode<T>>, args: &mut SearchArgs<'a, T, A>) -> Vec<SearchNode<T>> where T: 'a + SearchNodeExtraData {
//    debug_assert!(args.beam_width >= starts.len());

    // store the search result, a map from id of node's value => node to eliminate duplicated result
    let mut results_id: HashSet<String> = Default::default();
    let mut results: ExploringNodes<T> = Default::default();

    // ##############################################
    // Add very first nodes to kick off BEAM SEARCH
    let mut current_exploring_nodes_id: HashSet<String> = Default::default();
    let mut current_exploring_nodes: ExploringNodes<T> = Default::default();

    for n in starts {
        if n.is_terminal() {
            results_id.insert(n.id.clone());
            results.push(n);
        } else {
            current_exploring_nodes_id.insert(n.id.clone());
            current_exploring_nodes.push(n);
        }
    }

    // ##############################################
    // Start BEAM SEARCH!!
    let mut iter_no = 0;
    let mut next_exploring_nodes;
    let mut prev_exploring_nodes = Default::default();

    while results.len() < args.n_results && current_exploring_nodes.len() > 0 {
        iter_no += 1;
        trace!("=== ({}) BEAMSEARCH ===> Doing round: {}", args.sm_id, iter_no);
        next_exploring_nodes = (*args.discover)(&current_exploring_nodes, &mut args.extra_args);
        // sort next nodes by their score, higher is better
        next_exploring_nodes.sort_by(args.compare_search_node);

        prev_exploring_nodes = current_exploring_nodes;
        current_exploring_nodes = Default::default();
        current_exploring_nodes_id = Default::default();
        for next_node in next_exploring_nodes.into_iter() {
            if next_node.is_terminal() || (*args.should_stop)(args, iter_no, &next_node) {
                if !results_id.contains(&next_node.id) {
                    results_id.insert(next_node.id.clone());
                    results.push(next_node);

                    if results.len() == args.n_results {
                        break;
                    }
                }
            } else {
                if !current_exploring_nodes_id.contains(&next_node.id) {
                    current_exploring_nodes_id.insert(next_node.id.clone());
                    current_exploring_nodes.push(next_node);
                    if current_exploring_nodes.len() == args.beam_width {
                        break;
                    }
                }
            }
        }

        if results.len() == args.n_results {
            break;
        }

        if current_exploring_nodes.len() > 0 {
            // log search nodes
            args.log_search_nodes(iter_no, &current_exploring_nodes);
        }
    }

    // ##############################################
    // Add more results to fulfill the requirements
    if results.len() == 0 {
        let best_node = if current_exploring_nodes.len() > 0 {
            // select best node by its scoring
            current_exploring_nodes.into_iter()
                .max_by(|x, y| x.score.partial_cmp(&y.score).unwrap()).expect("Not empty")
        } else {
            if prev_exploring_nodes.len() == 0 {
                return Vec::new();
            }
            prev_exploring_nodes.into_iter()
                .max_by(|x, y| x.score.partial_cmp(&y.score).unwrap()).expect("Not empty")
        };

        return vec![best_node];
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results
}
