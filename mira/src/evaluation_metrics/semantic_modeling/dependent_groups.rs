use evaluation_metrics::semantic_modeling::internal_structure::*;
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp;
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use std::fmt;
use std::iter::FromIterator;
use permutohedron::factorial;

#[derive(PartialEq, Eq)]
pub struct DependentGroups<'a> {
    pub pair_groups: Vec<PairLabelGroup<'a>>,
    pub x_triples: HashSet<Triple<'a>>,
    pub x_prime_triples: HashSet<Triple<'a>>
}

impl<'a> DependentGroups<'a> {
    pub fn new(pair_groups: Vec<PairLabelGroup<'a>>) -> DependentGroups<'a> {
        let x_triples: HashSet<Triple<'a>> = pair_groups.iter().flat_map(|g| g.x.get_triples()).collect();
        let x_prime_triples: HashSet<Triple<'a>> = pair_groups.iter().flat_map(|g| g.x_prime.get_triples()).collect();

        DependentGroups { pair_groups, x_triples, x_prime_triples }
    }

    pub fn get_n_permutations(&self) -> usize {
        let mut n_permutation = 1.0;
        for pair_group in &self.pair_groups {
            let n = cmp::max(pair_group.x.size(), pair_group.x_prime.size());
            let m = cmp::min(pair_group.x.size(), pair_group.x_prime.size());

            n_permutation *= factorial(n) as f64 / factorial(n - m) as f64;
        }

        n_permutation as usize
    }

    pub fn extends(&mut self, pair_groups: Vec<PairLabelGroup<'a>>) {
        for pair_group in &pair_groups {
            self.x_triples.extend(pair_group.x.get_triples());
            self.x_prime_triples.extend(pair_group.x_prime.get_triples());
        }

        self.pair_groups.extend(pair_groups);
    }

    /// This method takes a list of groups (X, X') and group them based on their dependencies.
    ///
    /// D = {D1, D2, …} s.t for all Di, Dj, (Xi, Xi') in Di, (Xj, Xj’) in Dj, they are independent
    /// Two groups of nodes are dependent when at least one unbounded nodes in a group is a label of other group.
    /// For example, "actor_appellation" has link to "type", so group "actor_appellation" depends on group "type"
    pub fn split_by_dependency(map_groups: Vec<PairLabelGroup<'a>>, bijection: &Bijection) -> Vec<DependentGroups<'a>> {
        let mut dependency_map: Vec<Vec<i32>> = vec![Vec::new(); map_groups.len()];
        let group_label2idx: HashMap<&'a String, i32> = map_groups.iter().enumerate()
            .map(|(i, u)| (u.label(), i as i32))
            .collect();

        for (i, map_group) in map_groups.iter().enumerate() {
            let common_labels = DependentGroups::get_common_unbounded_nodes(&map_group.x, &map_group.x_prime, bijection);
            for common_label in common_labels {
                dependency_map[i].push(group_label2idx[common_label]);
            }
        }

        let dependency_groups = DependentGroups::group_dependent_elements(&dependency_map);

        let mut dependency_pair_groups: FnvHashMap<i32, Vec<PairLabelGroup<'a>>> = Default::default();
        for (i, map_group) in map_groups.into_iter().enumerate() {
            dependency_pair_groups.entry(dependency_groups[i]).or_insert(Vec::new()).push(map_group);
        }

        dependency_pair_groups.into_iter()
            .map(|(_k, v)| DependentGroups::new(v)).collect()
    }

    pub fn combine(dependent_groups: Vec<DependentGroups<'a>>) -> DependentGroups<'a> {
        let mut pair_group = Vec::with_capacity(dependent_groups.iter().map(|g| g.pair_groups.len()).sum());

        for mut group in dependent_groups {
            pair_group.append(&mut group.pair_groups);
        }

        DependentGroups::new(pair_group)
    }

    /// algorithm to merge the dependencies
    /// input:
    ///     - dependency_map: [<element_index, ...>, ...] list of dependencies where element at ith
    ///             position is list of index of elements that element at ith position depends upon.
    /// output:
    ///     - dependency_groups: [<group_id>, ...] list of group id, where element at ith position
    ///             is group id that element belongs to
    /// e.g:
    fn group_dependent_elements(dependency_map: &Vec<Vec<i32>>) -> Vec<i32> {
        // TODO: improve this function
        let mut dependency_groups = vec![-1; dependency_map.len()];
        let mut invert_dependency_groups: FnvHashMap<i32, FnvHashSet<i32>> = Default::default();

        for (i, g) in dependency_map.iter().enumerate() {
            let mut groups: FnvHashSet<i32> = FnvHashSet::from_iter(g.iter().map(|&j| dependency_groups[j as usize]));
            groups.insert(dependency_groups[i]);
            let has_unbounded_elements = groups.remove(&-1);
            let group_id: i32;

            if groups.len() == 0 {
                group_id = invert_dependency_groups.len() as i32;
                invert_dependency_groups.insert(group_id, Default::default());
            } else {
                group_id = groups.iter().next().unwrap().clone()
            };

            if has_unbounded_elements {
                // map unbounded elements to group has group_id
                for j in g {
                    if dependency_groups[*j as usize] == -1 {
                        dependency_groups[*j as usize] = group_id;
                        invert_dependency_groups.get_mut(&group_id).unwrap().insert(*j);
                    }
                }
            }

            let modified = invert_dependency_groups.get_mut(&group_id).unwrap() as *mut FnvHashSet<i32>;
            for another_group_id in &groups {
                if group_id != *another_group_id {
                    for j in &invert_dependency_groups[another_group_id] {
                        dependency_groups[*j as usize] = group_id;
                        unsafe { (&mut *modified).insert(*j); }
                    }
                }
            }
        }

        dependency_groups
    }

    /// Finding unbounded nodes in X and X_prime that have same labels
    fn get_common_unbounded_nodes(x: &LabelGroup<'a>, x_prime: &LabelGroup<'a>, bijection: &Bijection)-> Vec<&'a String> {
        let unbounded_x = DependentGroups::get_unbounded_labels(x, |nid| bijection.is_gold_node_bounded(nid));
        let unbounded_x_prime = DependentGroups::get_unbounded_labels(x_prime, |nid| bijection.is_pred_node_bounded(nid));

        HashSet::<&String>::from_iter(unbounded_x.into_iter())
            .intersection(&HashSet::<&String>::from_iter(unbounded_x_prime.into_iter()))
            .into_iter().map(|s| *s).collect()
    }

    /// Get nodes of a label group which have not been bounded by a bijection
    fn get_unbounded_labels<F>(group: &LabelGroup<'a>, is_bounded_func: F) -> Vec<&'a String>
       where F: Fn(usize) -> bool {
        let mut unbounded_nodes = Vec::new();

        for n in &group.nodes {
            for e in n.iter_incoming_edges(group.graph) {
                if !is_bounded_func(e.source_id) {
                    unbounded_nodes.push(&e.get_source_node(group.graph).label);
                }
            }

            for e in n.iter_outgoing_edges(group.graph) {
                if !is_bounded_func(e.target_id) {
                    unbounded_nodes.push(&e.get_target_node(group.graph).label);
                }
            }
        }

        unbounded_nodes
    }
}

impl<'a> fmt::Debug for DependentGroups<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "DependentGroups [")?;
        for group in self.pair_groups.iter() {
            write!(f, "{} ({}-{}),", group.label(), group.x.nodes.len(), group.x_prime.nodes.len())?;
        }
        write!(f, "]")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use super::super::alignment::tests::*;
    use algorithm::prelude::*;
    use evaluation_metrics::semantic_modeling::*;

    pub fn test_get_common_unbounded_nodes() {
    }

    #[test]
    pub fn test_split_by_dependency() {
        let (gold_sm, gold_idmap) = quick_graph(&[
            "ManMadeObject1--was_classified_by--TypeAssignment1",
            "ManMadeObject1--was_classified_by--TypeAssignment2",
            "TypeAssignment1--assigned--Type1",
            "TypeAssignment2--assigned--Type2",
            "Type1--label--type_a",
            "Type2--label--type_b",
        ]);

        let (pred_sm, pred_idmap) = quick_graph(&[
            "ManMadeObject1--was_classified_by--TypeAssignment1",
            "TypeAssignment1--assigned--Type1",
            "Type1--label--type_b",
        ]);

        let mut bijection = Bijection::new(gold_sm.n_nodes, pred_sm.n_nodes);
        bijection.push_both(0, 0);
        bijection.push_x(5, None);
        bijection.push_both(6, 3);

        let map_groups = vec![
            PairLabelGroup {
                x: LabelGroup::new(&gold_sm, get_nodes(&gold_sm, &[1, 2]), DataNodeMode::NoTouch),
                x_prime: LabelGroup::new(&pred_sm, get_nodes(&pred_sm, &[1]), DataNodeMode::NoTouch),
            },
            PairLabelGroup {
                x: LabelGroup::new(&gold_sm, get_nodes(&gold_sm, &[3, 4]), DataNodeMode::NoTouch),
                x_prime: LabelGroup::new(&pred_sm, get_nodes(&pred_sm, &[2]), DataNodeMode::NoTouch),
            },
        ];

        let groups = DependentGroups::split_by_dependency(map_groups.clone(), &bijection);
        assert_eq!(groups, vec![DependentGroups::new(map_groups)]);
    }
}