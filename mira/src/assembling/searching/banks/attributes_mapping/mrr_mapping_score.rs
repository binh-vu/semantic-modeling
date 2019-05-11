use assembling::searching::banks::data_structure::int_graph::IntGraph;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::MappingCandidate;
use assembling::models::mrr::MRRModel;
use algorithm::data_structure::graph::*;
use fnv::FnvHashMap;
use rdb2rdf::models::semantic_model::SemanticModel;


pub fn mrr_mapping_score(mrr_model: &MRRModel, sm: &SemanticModel, int_graph: &IntGraph, attr_mappings: &[MappingCandidate]) -> Vec<f64> {
    // re_construct a graph from those attr_mapping
    let graphs = attr_mappings.iter()
        .map(|mc| {
            let mut g = Graph::with_capacity("".to_owned(), attr_mappings.len() * 2, attr_mappings.len(), true, true, true);
            let mut idmap: FnvHashMap<usize, usize> = Default::default();

            for &(attr_id, (st, ie)) in mc.mapping.iter() {
                let ie_source = int_graph.graph.get_node_by_id(ie.source_id);
                if !idmap.contains_key(&ie.source_id) {
                    idmap.insert(ie.source_id, g.add_node(Node::new(NodeType::ClassNode, ie_source.label.clone())));
                }

                idmap.insert(ie.target_id, g.add_node(Node::new(NodeType::DataNode, sm.graph.get_node_by_id(attr_id).label.clone())));
                g.add_edge(Edge::new(EdgeType::Unspecified, ie.label.clone(), idmap[&ie.source_id], idmap[&ie.target_id]));
            }
            g
        })
        .collect::<Vec<_>>();

    mrr_model.predict_sm_probs(&sm.id, graphs)
        .iter()
        .map(|(g, s)| *s)
        .collect()
}