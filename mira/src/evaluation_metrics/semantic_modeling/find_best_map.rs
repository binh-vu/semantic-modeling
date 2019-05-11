use algorithm::data_structure::graph::*;
use std::collections::HashMap;
use im::vector::Vector as IVector;

use super::internal_structure::*;
use super::dependent_groups::*;
use std::cmp;
use std::collections::HashSet;
use permutohedron::heap_recursive;
use itertools::Itertools;
use fnv::FnvHashSet;

pub struct FindBestMapArgs {
    pub group_index: usize,
    pub bijection: Bijection
}

pub struct IterGroupMapUsingGroupingArgs {
    pub node_index: usize,
    pub group_sizes: IVector<i32>,
    pub mapping: IVector<i32> // originally named bijection in python code
}

pub fn iter_group_maps<F>(x: &LabelGroup, x_prime: &LabelGroup, borrow_bijection: &mut Bijection, func: F)
    where F: FnMut(&Bijection) -> () {
    // note that the borrow bijection must be cleaned before using
    if x.size() < x_prime.size() {
        iter_group_maps_general_approach(x, x_prime, borrow_bijection, func)
    } else {
        iter_group_maps_using_grouping(x_prime, x.group_by_structures(x_prime), borrow_bijection, func)
    }
}

/// Generate all mapping from X to X_prime
/// NOTE: |X| < |X_prime|
///
/// Return mapping from (x_prime to x)
fn iter_group_maps_general_approach<F>(x: &LabelGroup, x_prime: &LabelGroup, borrow_bijection: &mut Bijection, mut func: F)
    where F: FnMut(&Bijection) -> () {

    let mut mapping_mold: Vec<Option<usize>> = vec![None; x_prime.size()];
    for mut combs in (0..x_prime.size()).combinations(x.size()) {
        heap_recursive(&mut combs, |perm| {
            for (i, &j) in perm.iter().enumerate() {
                mapping_mold[j] = Some(x.nodes[i].id);
            }

            for i in 0..x_prime.size() {
                borrow_bijection.push_x_prime(mapping_mold[i], x_prime.nodes[i].id);
                mapping_mold[i] = None;
            }

            func(&borrow_bijection);
            borrow_bijection.clear(); // clean it after using
        });
    }
}

/// Generate all mapping from X_prime to G (nodes in X grouped by their structures)
/// NOTE: |X_prime| <= |X|
///
/// Return mapping from (x_prime to x)
fn iter_group_maps_using_grouping<'a, F>(x_prime: &LabelGroup<'a>, x_groups: Vec<StructureGroup<'a>>, borrow_bijection: &mut Bijection, mut func: F)
    where F: FnMut(&Bijection) -> () {
    let terminate_index = x_prime.size();
    let mut call_stack = vec![IterGroupMapUsingGroupingArgs {
        node_index: 0,
        mapping: IVector::from(vec![-1; x_prime.size()]),
        group_sizes: IVector::from(x_groups.iter().map(|g| g.size() as i32).collect::<Vec<_>>())
    }];
    let mut g_numerator = vec![0; x_groups.len()];

    loop {
        let call_args = call_stack.pop().unwrap();
        if call_args.node_index == terminate_index {
            // convert bijection into final mapping
            for i in 0..x_prime.size() {
                let mapping_idx = call_args.mapping[i] as usize;
                let x_prime_id = x_prime.nodes[i].id;
                let x_id = x_groups[mapping_idx].nodes[g_numerator[mapping_idx]].id;

                g_numerator[mapping_idx] += 1;
                borrow_bijection.push_both(x_id, x_prime_id)
            }

            func(&borrow_bijection);
            for i in 0..x_groups.len() {
                g_numerator[i] = 0;
            }
            borrow_bijection.clear();
        } else {
            for i in 0..x_groups.len() {
                if call_args.group_sizes[i] == 0 {
                    continue;
                }

                let mapping = call_args.mapping.update(call_args.node_index, i as i32);
                let group_sizes = call_args.group_sizes.update(i, *call_args.group_sizes.get(i).unwrap() - 1);
                call_stack.push(IterGroupMapUsingGroupingArgs {
                    node_index: call_args.node_index + 1,
                    mapping,
                    group_sizes
                });
            }
        }

        if call_stack.len() == 0 {
            break;
        }
    }
}

pub fn eval_score<'a>(dependent_group: &DependentGroups<'a>, bijection: &Bijection) -> f64 {
    let mut x_prime_triples: HashSet<Triple<'a>> = Default::default();

    for triple in &dependent_group.x_prime_triples {
        let new_triple = Triple {
            source_id: bijection.prime2x[triple.source_id as usize],
            predicate: triple.predicate,
            target_id: if triple.target_id == DEFAULT_IGNORE_LABEL_DATA_NODE_ID {
                triple.target_id
            } else {
                bijection.to_x(triple.target_id as usize)
            }
        };

        x_prime_triples.insert(new_triple);
    }

    dependent_group.x_triples.intersection(&x_prime_triples).count() as f64
}

pub fn find_best_map<'a>(dependent_group: &DependentGroups<'a>, bijection: Bijection) -> Bijection {
    let terminate_index = dependent_group.pair_groups.len();
    // a lending bijection to help speed up the program by avoiding creating new bijection
    // everytime
    let mut lending_bijection = Bijection::new_like(&bijection);

    let mut call_stack = vec![FindBestMapArgs { group_index: 0, bijection }];
    let mut best_score = -1.0;
    let mut best_map: Bijection = Bijection::empty();

    while call_stack.len() > 0 {
        let call_args = call_stack.pop().unwrap();

        if call_args.group_index == terminate_index {
            // it is terminated, calculate score
            let score = eval_score(dependent_group, &call_args.bijection);
            if score > best_score {
                best_score = score;
                best_map = call_args.bijection;
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

    best_map
}