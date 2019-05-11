use assembling::searching::banks::data_structure::int_graph::IntGraph;
use rdb2rdf::models::semantic_model::SemanticModel;
use assembling::models::example::MRRExample;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::generate_candidate_attr_mapping;
use assembling::searching::banks::attributes_mapping::mapping_score::mohsen_mapping_score;
use assembling::searching::banks::attributes_mapping::learned_mapping_score::compute_gold_mapping;
use assembling::searching::banks::attributes_mapping::eval_mapping_score::evaluate_attr_mapping;
use fnv::FnvHashMap;
use algorithm::data_structure::graph::*;
use rdb2rdf::models::semantic_model::SemanticType;
use assembling::searching::banks::data_structure::int_graph::IntEdge;
use assembling::searching::banks::attributes_mapping::learned_mapping_score::repopulate_attr_mapping;
use assembling::auto_label::MRRLabel;
use im::HashMap as IHashMap;
use assembling::models::annotator::Annotator;
use rayon::prelude::*;
use assembling::auto_label;

pub fn get_mrr_train_data(annotator: &Annotator, int_graph: &IntGraph, train_sms: &[&SemanticModel], test_sms: &[&SemanticModel]) -> (Vec<MRRExample>, Vec<MRRExample>) {
    let branching_factor = 50;
    let mut train_examples = train_sms.par_iter()
        .flat_map(|sm| {
            let mut attr_mappings = generate_candidate_attr_mapping(int_graph, &sm.attrs, branching_factor, &mut mohsen_mapping_score);
            attr_mappings.push(make_oracle_mapping(int_graph, sm));

            attr_mappings.iter()
                .map(|attr_mapping| make_label_example(annotator, int_graph, sm, attr_mapping))
                .collect::<Vec<_>>()
        })
        .filter(|e| e.is_some())
        .map(|e| e.unwrap())
        .collect::<Vec<_>>();

   let mut test_examples = test_sms.par_iter()
       .flat_map(|sm| {
           let mut attr_mappings = generate_candidate_attr_mapping(int_graph, &sm.attrs, branching_factor, &mut mohsen_mapping_score);
           attr_mappings.iter()
               .map(|attr_mapping| make_label_example(annotator, int_graph, sm, attr_mapping))
               .collect::<Vec<_>>()
       })
       .filter(|e| e.is_some())
       .map(|e| e.unwrap())
       .collect::<Vec<_>>();

    (train_examples, test_examples)
}

fn make_label_example(annotator: &Annotator, int_graph: &IntGraph, sm: &SemanticModel, attr_mapping: &FnvHashMap<usize, usize>) -> Option<MRRExample> {
    let attr_mapping = repopulate_attr_mapping(int_graph, sm, attr_mapping);
    let g = attr_mapping_to_graph(int_graph, sm, &attr_mapping);

    annotator.create_labeled_mrr_example(&sm.id, g)
}

fn attr_mapping_to_graph(int_graph: &IntGraph, sm: &SemanticModel, attr_mapping: &IHashMap<usize, (&SemanticType, &IntEdge)>) -> Graph {
    let mut g = Graph::with_capacity(sm.id.clone(), attr_mapping.len() * 2, attr_mapping.len(), true, true, true);
    let mut idmap: FnvHashMap<usize, usize> = Default::default();

    for &(attr_id, (st, ie)) in attr_mapping.iter() {
        let ie_source = int_graph.graph.get_node_by_id(ie.source_id);
        if !idmap.contains_key(&ie.source_id) {
            idmap.insert(ie.source_id, g.add_node(Node::new(NodeType::ClassNode, ie_source.label.clone())));
        }

        idmap.insert(ie.target_id, g.add_node(Node::new(NodeType::DataNode, sm.graph.get_node_by_id(attr_id).label.clone())));
        g.add_edge(Edge::new(EdgeType::Unspecified, ie.label.clone(), idmap[&ie.source_id], idmap[&ie.target_id]));
    }

    for n in g.iter_data_nodes() {
        if n.n_outgoing_edges > 0 {
            println!("[DEBUG] idmap = {:?}", idmap);
            println!("WTF");
        }
    }

    g
}

pub fn make_oracle_mapping(int_graph: &IntGraph, sm: &SemanticModel) -> FnvHashMap<usize, usize> {
    // create a subgraph of int_graph that only contains nodes & edges in sm
    let mut sub_int_graph = Graph::new("".to_owned(), true, true, false);
    let mut node_idmap: FnvHashMap<usize, usize> = Default::default();
    let mut edge_idmap: Vec<usize> = Default::default();

    for n in int_graph.graph.iter_nodes() {
        if n.data.tags.contains(&sm.id) {
            if n.is_data_node() {
                node_idmap.insert(n.id, sub_int_graph.add_node(Node::new(n.kind.clone(), n.data.original_names[&sm.id].clone())));
            } else {
                node_idmap.insert(n.id, sub_int_graph.add_node(Node::new(n.kind.clone(), n.label.clone())));
            }
        }
    }

    for e in int_graph.graph.iter_edges() {
        if e.data.tags.contains(&sm.id) {
            sub_int_graph.add_edge(Edge::new(e.kind.clone(), e.label.clone(), node_idmap[&e.source_id], node_idmap[&e.target_id]));
            edge_idmap.push(e.id);
        }
    }

    // align the subgraph with sm graph
    let mrr_label = auto_label::label_no_ambiguous(&sm.graph, &sub_int_graph, 60000).unwrap();
    debug_assert_eq!(mrr_label.f1, 1.0);
    
    let mut mapping: FnvHashMap<usize, usize> = Default::default();

    for attr in sm.attrs.iter() {
        let dnode_id = mrr_label.bijection.x2prime[attr.id] as usize;
        let int_edge = sub_int_graph.get_node_by_id(dnode_id).first_incoming_edge(&sub_int_graph).unwrap();

        mapping.insert(attr.id, edge_idmap[int_edge.id]);
    }

    mapping
}