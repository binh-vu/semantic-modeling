use evaluation_metrics::semantic_modeling::alignment::*;
use evaluation_metrics::semantic_modeling::find_best_map::*;
use evaluation_metrics::semantic_modeling::internal_structure::*;
use evaluation_metrics::semantic_modeling::dependent_groups::*;
use algorithm::data_structure::graph::Graph;
use algorithm::combination::IterIndex;
use std::cmp;

pub fn align_graph<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode, max_permutation: usize) -> Vec<(f64, f64, f64, Bijection, TripleSet<'a>)> {
    let (mut bijection, dependent_groups, independent_groups) = get_dependent_groups(gold_sm, pred_sm, data_node_mode);
    let n_permutation: usize = dependent_groups.iter().map(|g| g.get_n_permutations()).sum();

    if n_permutation > max_permutation {
        debug!("Ignore this example because number of permutation = {} > {}", n_permutation, max_permutation);
        return Vec::new();
    }

    // === MODIFIED START HERE ===
    let partial_bijection = bijection;
    let mut best_sub_bijections = Vec::with_capacity(dependent_groups.len());
    for dependent_group in &dependent_groups {
        best_sub_bijections.push(find_all_best_map(dependent_group, partial_bijection.clone()));
    }

    let mut bijections = Vec::new();
    for best_bijection_idxs in IterIndex::new(best_sub_bijections.iter().map(|array| array.len()).collect()).iter() {
        let mut bijection = partial_bijection.clone();
        for (i, &idx) in best_bijection_idxs.iter().enumerate() {
            bijection.extends_(&best_sub_bijections[i][idx]);
        }
        bijections.push(bijection);
    }

    let bijection = bijections.pop().unwrap();
    // === MODIFIED END HERE ===

    let mut all_groups = DependentGroups::combine(dependent_groups);
    all_groups.extends(independent_groups);
    let single_res = compute_results(all_groups, bijection);

    let mut results = Vec::new();
    for bijection in bijections {
        results.push((single_res.0, single_res.1, single_res.2, bijection, single_res.4.clone()));
    }
    results.push(single_res);

    results
}

pub fn align_graph_one<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode, max_permutation: usize) -> Option<(f64, f64, f64, Bijection, TripleSet<'a>)> {
    let (mut bijection, dependent_groups, independent_groups) = get_dependent_groups(gold_sm, pred_sm, data_node_mode);
    let n_permutation: usize = dependent_groups.iter().map(|g| g.get_n_permutations()).sum();

    if n_permutation > max_permutation {
        debug!("Ignore this example because number of permutation = {} > {}", n_permutation, max_permutation);
        return None;
    }

    for dependent_group in &dependent_groups {
        bijection = find_best_map(dependent_group, bijection);
    }

    let mut all_groups = DependentGroups::combine(dependent_groups);
    all_groups.extends(independent_groups);
    Some(compute_results(all_groups, bijection))
}

pub fn align_graph_no_ambiguous<'a>(gold_sm: &'a Graph, pred_sm: &'a Graph, data_node_mode: DataNodeMode, max_permutation: usize) -> Option<(f64, f64, f64, Bijection, TripleSet<'a>)> {
    let (mut bijection, dependent_groups, independent_groups) = get_dependent_groups(gold_sm, pred_sm, data_node_mode);
    let n_permutation: usize = dependent_groups.iter().map(|g| g.get_n_permutations()).sum();

    if n_permutation > max_permutation {
        debug!("Ignore this example because number of permutation = {} > {}", n_permutation, max_permutation);
        return None;
    }

    // === MODIFIED START HERE ===
    for dependent_group in &dependent_groups {
        let mut bijections = find_all_best_map(dependent_group, bijection);
        if bijections.len() == 1 {
            bijection = bijections.pop().unwrap();
        } else {
            return None;
        }
    }
    // === MODIFIED END HERE ===

    let mut all_groups = DependentGroups::combine(dependent_groups);
    all_groups.extends(independent_groups);
    Some(compute_results(all_groups, bijection))
}

fn find_all_best_map(dependent_group: &DependentGroups, bijection: Bijection) -> Vec<Bijection> {
    let terminate_index = dependent_group.pair_groups.len();
    // a lending bijection to help speed up the program by avoiding creating new bijection
    // everytime
    let mut lending_bijection = Bijection::new_like(&bijection);

    let mut call_stack = vec![FindBestMapArgs { group_index: 0, bijection }];
    let mut best_score = -1.0;
    let mut best_maps = Vec::new();

    while call_stack.len() > 0 {
        let call_args = call_stack.pop().unwrap();

        if call_args.group_index == terminate_index {
            // it is terminated, calculate score
            let score = eval_score(dependent_group, &call_args.bijection);
            if score > best_score {
                best_score = score;
                best_maps.clear();
                best_maps.push(call_args.bijection);
            } else if score == best_score  {
                best_maps.push(call_args.bijection);
            }
        } else {
            let pair_group = &dependent_group.pair_groups[call_args.group_index];
            iter_group_maps(&pair_group.x, &pair_group.x_prime, &mut lending_bijection, |par_bijection| {
               call_stack.push(FindBestMapArgs {
                   group_index: call_args.group_index + 1,
                   bijection: call_args.bijection.extends(par_bijection)
               });
            });
        }
    }

    best_maps
}

#[inline]
fn compute_results(all_groups: DependentGroups, bijection: Bijection) -> (f64, f64, f64, Bijection, TripleSet) {
    let tp = eval_score(&all_groups, &bijection);
    let recall = tp / cmp::max(all_groups.x_triples.len(), 1) as f64;
    let precision = tp / cmp::max(all_groups.x_prime_triples.len(), 1) as f64;
    let f1 = if tp == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };

    (f1, precision, recall, bijection, all_groups.x_triples)
}