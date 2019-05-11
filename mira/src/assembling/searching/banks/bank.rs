use rdb2rdf::models::semantic_model::SemanticModel;
use assembling::searching::banks::data_structure::int_graph::IntGraph;
use std::collections::HashMap;
use algorithm::data_structure::graph::Graph;
use rdb2rdf::ontology::ont_graph::OntGraph;
use assembling::searching::banks::attributes_mapping::mapping_score::mohsen_mapping_score;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::generate_candidate_attr_mapping;
use algorithm::data_structure::graph::*;
use std::collections::binary_heap::BinaryHeap;
use std::collections::HashSet;
use assembling::searching::banks::data_structure::sssp::DijkstraSSSPIterator;
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use assembling::searching::banks::data_structure::int_graph::*;
use std::cmp::Ordering;
use assembling::searching::banks::data_structure::search_node::BankCandidateSM;
use itertools::Itertools;
use assembling::searching::banks::attributes_mapping::eval_mapping_score::evaluate_attr_mapping;
use rdb2rdf::models::semantic_model::SemanticType;
use im::HashMap as IHashMap;
use assembling::searching::banks::attributes_mapping::generate_candidate_attr_mapping::MappingCandidate;

#[derive(PartialEq)]
pub struct State {
    cost: f32,
    // index of DijkstraSSSPIterator in the array
    sssp_index: usize,
    position: usize
}


impl State {
    pub fn new(cost: f32, sssp_index: usize) -> State {
        State {
            cost, sssp_index, position: 0
        }
    }
}

impl Eq for State {}


impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap()
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &State) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


/// Return a list of steiner trees and it's node id mapping
pub fn banks(weighted_graph: &Graph<INodeData, IEdgeData>, terminals: &[usize], top_k: usize) -> Vec<(Graph, FnvHashMap<usize, usize>)> {
    let mut results = Vec::with_capacity(top_k);
    let mut sssp_iterators: Vec<DijkstraSSSPIterator> = Vec::with_capacity(terminals.len());
    let mut iterator_heap: BinaryHeap<_> = BinaryHeap::new();
    let mut visiting_info: FnvHashMap<usize, FnvHashSet<usize>> = Default::default();

    for &term_id in terminals {
        let sssp = DijkstraSSSPIterator::new(weighted_graph, term_id);
        iterator_heap.push(State::new(sssp.distance2nextnode(), sssp_iterators.len()));
        sssp_iterators.push(sssp);
    }

    while iterator_heap.len() > 0 && results.len() < top_k {
        let state = iterator_heap.pop().unwrap();
        let node_id = sssp_iterators[state.sssp_index].travel_next_node();
        let sssp = &sssp_iterators[state.sssp_index];

        if !sssp.is_finish() {
            iterator_heap.push(State::new(sssp.distance2nextnode(), state.sssp_index));
        }

        if !visiting_info.contains_key(&node_id) {
            visiting_info.insert(node_id, Default::default());
        }

        visiting_info.get_mut(&node_id).unwrap().insert(sssp.source_id);
        if visiting_info[&node_id].len() == terminals.len() {
            // it forms a connection tree rooted at node v
            let mut tree = Graph::new("".to_owned(), true, true, false);
            let mut id_map: FnvHashMap<usize, usize> = Default::default();
            for sssp in &sssp_iterators {
                let forward_path = sssp.get_shortest_path(node_id);
                if !id_map.contains_key(&node_id) {
                    let gn = weighted_graph.get_node_by_id(node_id);
                    id_map.insert(node_id, tree.add_node(Node::new(gn.kind.clone(), gn.label.clone())));
                }

                for p in &forward_path {
                    if !id_map.contains_key(&p.node_id) {
                        let gn = weighted_graph.get_node_by_id(p.node_id);
                        id_map.insert(p.node_id, tree.add_node(Node::new(gn.kind.clone(), gn.label.clone())));
                    }
                }

                for p in &forward_path {
                    let ge = weighted_graph.get_edge_by_id(p.outgoing_edge_id);
                    // don't need to double check because we don't have duplicated edge
                    tree.add_edge(Edge::new(EdgeType::Unspecified, ge.label.clone(), id_map[&ge.source_id], id_map[&ge.target_id]));
                }
            }
            results.push((tree, id_map));
        }
    }

    results
}


/// Reverse all edges in the graph
pub fn reversed<ND: NodeData, ED: EdgeData>(g: &Graph<ND, ED>) -> Graph<ND, ED> {
    let mut rg = Graph::new_like(g);
    for n in g.iter_nodes() {
        rg.add_node(Node::new(n.kind.clone(), n.label.clone()));
    }

    for e in g.iter_edges() {
        rg.add_edge(Edge::new(e.kind.clone(), e.label.clone(), e.target_id, e.source_id));
    }

    return rg;
}


pub fn generate_candidate_sms<'a, F>(weighted_graph: &'a IntGraph, sm: &SemanticModel, func: &mut F) -> Vec<BankCandidateSM<'a>>
    where F: FnMut(&IntGraph, &[MappingCandidate]) -> Vec<f64> {

    let branching_factor = 50;
    let top_k = 10;
    let mut sm_candidates = HashMap::new();

    let reversed_weighted_graph = reversed(&weighted_graph.graph);

    let candidate_attr_mappings = generate_candidate_attr_mapping(weighted_graph, &sm.attrs, branching_factor, func);
    info!("#candidates mappings: {}", candidate_attr_mappings.len());

    let mut possible_terminals: HashMap<Vec<usize>, Vec<FnvHashMap<usize, usize>>> = Default::default();
    for attr_mapping in candidate_attr_mappings {
        let mut internal_nodes = attr_mapping.values()
            .map(|&eid| weighted_graph.graph.get_edge_by_id(eid).source_id)
            .unique()
            .collect::<Vec<_>>();
        internal_nodes.sort();
        possible_terminals.entry(internal_nodes).or_insert(Default::default()).push(attr_mapping);
    }

    for (terminals, attr_mappings) in possible_terminals {
        let steiner_trees = banks(&reversed_weighted_graph, &terminals, top_k);

        for (tree, node_id_map) in &steiner_trees {
            let mut rg = reversed(tree);

            // now we add data nodes to finish the final graphs
            for attr_mapping in &attr_mappings {
                let mut g = rg.clone();
                let mut edge_idmap = Vec::new();

                for (&attr_id, &eid) in attr_mapping {
                    let dnode_id = g.add_node(Node::new(NodeType::DataNode, sm.graph.get_node_by_id(attr_id).label.clone()));
                    let e = weighted_graph.graph.get_edge_by_id(eid);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), node_id_map[&e.source_id], dnode_id));
                    edge_idmap.push(eid);
                }

                sm_candidates.insert(get_acyclic_consistent_unique_hashing(&g), BankCandidateSM::new(g, edge_idmap, weighted_graph));
            }
        }
    }

    let mut results = sm_candidates.into_iter().map(|(k, v)| v).collect::<Vec<_>>();
    // sorted by coherence like in mohsen papers
    results.sort_by(|a, b| b.mohsen_coherence_score.partial_cmp(&a.mohsen_coherence_score).unwrap());
    results
}