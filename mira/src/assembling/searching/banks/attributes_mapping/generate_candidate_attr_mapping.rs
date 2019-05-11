use im::HashMap as IHashMap;
use im::HashSet as IHashSet;
use rdb2rdf::models::semantic_model::SemanticType;
use rdb2rdf::models::semantic_model::Attribute;
use std::collections::HashMap;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use assembling::searching::banks::data_structure::int_graph::IntEdge;
use fnv::FnvHashMap;
use fnv::FnvHashSet;


#[derive(Debug)]
pub struct MappingCandidate<'a> {
    pub mapping: IHashMap<usize, (&'a SemanticType, &'a IntEdge)>,
    used_edges: IHashSet<usize>,
    score: f64
}


/// Generate candidates of mapping from attribute to node id in the model graph (beam search).
///    Note: If the provided semantic types contain duplicate pair (source/domain & type), then output mappings
///    will contain duplicated attribute mapping
///
/// Output of this function is a list of candidate attribute mappings: from attribute id => edge id in the int_graph
pub fn generate_candidate_attr_mapping<F>(int_graph: &IntGraph, attrs: &[Attribute], branching_factor: usize, mapping_scoring_func: &mut F) -> Vec<FnvHashMap<usize, usize>>
    where F: FnMut(&IntGraph, &[MappingCandidate]) -> Vec<f64> {
    let mut mapping_candidates: Vec<MappingCandidate> = Default::default();

    for attr in attrs.iter() {
        let mut attr_mapping: Vec<_> = Default::default();
        for stype in &attr.semantic_types {
            for n in int_graph.graph.iter_nodes_by_label(&stype.class_uri) {
                for e in n.iter_outgoing_edges(&int_graph.graph) {
                    if e.label == stype.predicate && e.get_target_node(&int_graph.graph).is_data_node() {
                        attr_mapping.push((stype, e));
                    }
                }
            }
        }

        let mut new_mapping_candidates: Vec<MappingCandidate> = Default::default();
        if mapping_candidates.len() == 0 {
            for mapping in attr_mapping.into_iter() {
                let new_mapping_candidate = IHashMap::singleton(attr.id, mapping);
                new_mapping_candidates.push(MappingCandidate {
                    score: 0.0,
                    mapping: new_mapping_candidate,
                    used_edges: IHashSet::singleton(mapping.1.id)
                });
            }
        } else {
            for mapping_candidate in &mapping_candidates {
                for mapping in attr_mapping.iter() {
                    if !mapping_candidate.used_edges.contains(&mapping.1.id) {
                        // check because two attributes cannot map to same edge!
                        let new_mapping = mapping_candidate.mapping.update(attr.id, mapping.clone());
                        let new_mapping_candidate = MappingCandidate {
                            score: 0.0,
                            mapping: new_mapping,
                            used_edges: mapping_candidate.used_edges.update(mapping.1.id),
                        };
                        new_mapping_candidates.push(new_mapping_candidate);
                    }
                }
            }
        }

        if new_mapping_candidates.len() > 0 {
            // compute the score here
            let new_mapping_candidates_scores = mapping_scoring_func(int_graph, &new_mapping_candidates);
            for i in 0..new_mapping_candidates.len() {
                new_mapping_candidates[i].score = new_mapping_candidates_scores[i];
            }

            new_mapping_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            new_mapping_candidates.truncate(branching_factor);
            mapping_candidates = new_mapping_candidates;
        }
    }

    mapping_candidates.iter()
        .map(|mapping| {
            mapping.mapping.iter().map(|&(k, m)| (k, m.1.id)).collect::<FnvHashMap<_, _>>()
        })
        .collect::<Vec<_>>()
}


pub fn generate_candidate_attr_mapping_unordered<F>(int_graph: &IntGraph, attrs: &[Attribute], branching_factor: usize, mapping_scoring_func: &mut F) -> Vec<FnvHashMap<usize, usize>>
    where F: FnMut(&IntGraph, &[MappingCandidate]) -> Vec<f64> {
    let mut mapping_candidates: Vec<MappingCandidate> = Default::default();

    for attr in attrs.iter() {
        let mut attr_mapping: Vec<_> = Default::default();
        for stype in &attr.semantic_types {
            for n in int_graph.graph.iter_nodes_by_label(&stype.class_uri) {
                for e in n.iter_outgoing_edges(&int_graph.graph) {
                    if e.label == stype.predicate && e.get_target_node(&int_graph.graph).is_data_node() {
                        attr_mapping.push((stype, e));
                    }
                }
            }
        }

        let mut new_mapping_candidates: Vec<MappingCandidate> = Default::default();
        if mapping_candidates.len() == 0 {
            for mapping in attr_mapping.into_iter() {
                let new_mapping_candidate = IHashMap::singleton(attr.id, mapping);
                new_mapping_candidates.push(MappingCandidate {
                    score: 0.0,
                    mapping: new_mapping_candidate,
                    used_edges: IHashSet::singleton(mapping.1.id)
                });
            }
        } else {
            for mapping_candidate in &mapping_candidates {
                for mapping in attr_mapping.iter() {
                    if !mapping_candidate.used_edges.contains(&mapping.1.id) {
                        // check because two attributes cannot map to same edge!
                        let new_mapping = mapping_candidate.mapping.update(attr.id, mapping.clone());
                        let new_mapping_candidate = MappingCandidate {
                            score: 0.0,
                            mapping: new_mapping,
                            used_edges: mapping_candidate.used_edges.update(mapping.1.id),
                        };
                        new_mapping_candidates.push(new_mapping_candidate);
                    }
                }
            }
        }

        if new_mapping_candidates.len() > 0 {
            // compute the score here
            let new_mapping_candidates_scores = mapping_scoring_func(int_graph, &new_mapping_candidates);
            for i in 0..new_mapping_candidates.len() {
                new_mapping_candidates[i].score = new_mapping_candidates_scores[i];
            }

            new_mapping_candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            new_mapping_candidates.truncate(branching_factor);
            mapping_candidates = new_mapping_candidates;
        }
    }

    mapping_candidates.iter()
        .map(|mapping| {
            mapping.mapping.iter().map(|&(k, m)| (k, m.1.id)).collect::<FnvHashMap<_, _>>()
        })
        .collect::<Vec<_>>()
}