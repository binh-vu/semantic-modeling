use im::HashMap as IHashMap;
use assembling::searching::banks::data_structure::int_graph::IntEdge;
use rdb2rdf::models::semantic_model::SemanticType;
use fnv::FnvHashSet;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use std::collections::HashMap;
use rdb2rdf::models::semantic_model::SemanticModel;
use fnv::FnvHashMap;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::MappingCandidate;


pub fn mohsen_mapping_score(int_graph: &IntGraph, attr_mappings: &[MappingCandidate]) -> Vec<f64> {
    attr_mappings.iter()
        .map(|mc| {
            let attr_mapping = &mc.mapping;

            let nodes: FnvHashSet<usize> = attr_mapping.values()
                .flat_map(|m| vec![m.1.source_id, m.1.target_id])
                .collect::<FnvHashSet<_>>();
            let n_nodes = nodes.len();
            let mut tags: HashMap<&str, FnvHashSet<usize>> = Default::default();

            for (stype, edge) in attr_mapping.values() {
                for tag in &edge.get_source_node(&int_graph.graph).data.tags {
                    tags.entry(&tag).or_insert(Default::default()).insert(edge.source_id);
                }

                for tag in &edge.get_target_node(&int_graph.graph).data.tags {
                    tags.entry(&tag).or_insert(Default::default()).insert(edge.target_id);
                }
            }

            let confidence = attr_mapping.values().map(|m| m.0.score).sum::<f32>() / attr_mapping.len() as f32;
            let coherence = tags.values().map(|x| x.len()).max().unwrap() as f32 / n_nodes as f32;
            let size_reduction = (2 * attr_mapping.len() - n_nodes) as f32 / attr_mapping.len() as f32;

            ((confidence + coherence + size_reduction * 0.5) / 3.0) as f64
        })
        .collect::<Vec<_>>()
}