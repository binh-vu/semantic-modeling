use super::*;
use algorithm::prelude::*;
use assembling::annotator::Annotator;
use assembling::searching::beam_search::*;
use im::HashSet as IHashSet;
use im::OrdSet as IOrdSet;
use rdb2rdf::prelude::*;
use itertools::zip;

pub type IntTreeSearchNode = SearchNode<IntTreeSearchNodeData>;

pub fn get_started_pk_nodes(sm: &SemanticModel, annotator: &Annotator, args: &IntTreeSearchArgs) -> Vec<IntTreeSearchNode> {
    let mut search_seeds = vec![];
    let terminals = sm.attrs.iter().map(|attr| attr.label.clone()).collect::<IOrdSet<_>>();

    // create graph & graph explorer for each attributes
    for attr in &sm.attrs {
        for stype in &attr.semantic_types {
            if annotator.primary_key.contains(&stype.class_uri) && annotator.primary_key.get_primary_key(&stype.class_uri) == &stype.predicate {
                // this is primary key, we start from here
                let mut g: Graph = Graph::new(sm.id.clone(), true, true, false);
                let source_id = g.add_node(Node::new(NodeType::ClassNode, stype.class_uri.clone()));
                let target_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));
                g.add_edge(Edge::new(EdgeType::Unspecified, stype.predicate.clone(), source_id, target_id));

                let mut bijections = find_all_mounts(&stype, &args.int_graph)
                    .into_iter()
                    .map(|mount| Bijection::from_pairs(vec![mount.edge.source_id, mount.edge.target_id]))
                    .collect::<Vec<_>>();

                if bijections.len() > 0 {
                    let extra_data = IntTreeSearchNodeData {
                        graph: g,
                        remained_terminals: terminals.without(&attr.label),
                        bijections,
                    };
                    let search_node = IntTreeSearchNode::new(extra_data.get_id(), stype.score as f64, extra_data);
                    search_seeds.push(search_node);
                }
            }
        }
    }

    return search_seeds;
}


pub fn discover<'a>(search_nodes: &ExploringNodes<IntTreeSearchNodeData>, args: &mut IntTreeSearchArgs<'a>) -> Vec<IntTreeSearchNode> {
    let mut new_graphs = UniqueArray::<Graph, String>::new();
    let mut new_bijections = Vec::new();
    let mut remained_terminals = Vec::new();

    for search_node in search_nodes.iter() {
        for bijection in &search_node.data.bijections {
            let subtree = IntSubGraph::new(&search_node.data.graph, bijection, &args.int_graph);

            for attr_lbl in search_node.data.remained_terminals.iter() {
                let attr = args.sm.get_attr_by_label(attr_lbl);
                for plan in MergePlan::find_merge_plans(&subtree, attr, &args) {
                    if !(*args.merge_plan_filter)(&search_node.data.graph, &plan, args) {
                        continue;
                    }

                    let (new_g, new_bijection) = plan.proceed(&subtree, attr, &args);
                    if !(*args.sm_filter)(&new_g, args) {
                        continue;
                    }

                    if new_graphs.push(get_acyclic_consistent_unique_hashing(&new_g), new_g) {
                        new_bijections.push(new_bijection);
                        remained_terminals.push(search_node.data.remained_terminals.without(attr_lbl));
                    }
                }
            }
        }
    }

    let graph_w_scores = (*args.prob_candidate_sms)(new_graphs.get_value(), &new_bijections, args);
    let mut next_nodes = izip!(graph_w_scores.into_iter(), new_bijections.into_iter(), remained_terminals.into_iter())
        .map(|((g, score), bijection, remained_terms)| {
            let node_data = IntTreeSearchNodeData {
                graph: g,
                remained_terminals: remained_terms,
                bijections: vec![bijection]
            };

            IntTreeSearchNode::new(node_data.get_id(), score, node_data)
        })
        .collect::<Vec<_>>();

    next_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    next_nodes.truncate(args.beam_width);
    next_nodes
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use super::super::tests::*;
    use std::path::*;
    use assembling::searching::banks::data_structure::int_graph::IntGraph;
    use assembling::learning::elimination::discovery::{get_started_eliminated_2nodes, convert_to_constraint_nodes};
    use assembling::searching::banks::MohsenWeightingSystem;
    use settings::conf_search::ConstraintSpace;

    #[test]
    pub fn test_get_started_pk_nodes() {
        let int_graph = load_int_graph();
        let input = load_input();
        let annotator = input.get_annotator();
        let g: Graph = quick_graph(&["artistURI::d", "objectURI::d"]).0;

        let sm = SemanticModel::new(String::new(), vec![
            get_attribute(0, "artistURI", &["crm:E39_Actor--karma:classLink"], 0.68),
            get_attribute(0, "objectURI", &["crm:E22_Man-Made_Object--karma:classLink"], 0.68),
        ], g);
        let args = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);
        let started_nodes = get_started_pk_nodes(&sm, &annotator, &args);

        assert_eq!(started_nodes.len(), 2);
        assert!((started_nodes[0].score - 0.68).abs() <= 1e-7);
        assert_eq!(started_nodes[0].data.bijections, vec![
            Bijection::from_pairs(vec![4, 17]),
            Bijection::from_pairs(vec![67, 78])
        ]);
        assert_eq!(started_nodes[1].data.bijections, vec![
            Bijection::from_pairs(vec![1, 33]),
        ]);
    }

    #[test]
    pub fn test_discover_next_nodes() {
        let mut int_graph = load_int_graph();
        let input = load_input();
        let annotator = input.get_annotator();
        MohsenWeightingSystem::new(&int_graph, &annotator.train_sms).weight(&mut int_graph);

        let g: Graph = quick_graph(&[
            "crm:E12_Production1--crm:P14_carried_out_by--crm:E39_Actor1",
            "crm:E12_Production1--karma:classLink--productionURI::d",
            "crm:E39_Actor1--karma:classLink--artistURI::d",
        ]).0;

        let sm = SemanticModel::new(String::new(),
            vec![
                get_attribute(0, "artistURI", &["crm:E39_Actor--karma:classLink"], 0.68),
                get_attribute(1, "objectURI", &["crm:E22_Man-Made_Object--karma:classLink"], 0.68),
            ],
            quick_graph(&["artistURI::d", "objectURI::d"]).0);

        let search_node_data = IntTreeSearchNodeData {
            graph: g,
            remained_terminals: IOrdSet::singleton("objectURI".to_owned()),
            bijections: vec![Bijection::from_pairs(vec![0, 4, 27, 17])],
        };
        let mut args = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);

        let search_node = SearchNode::new(search_node_data.get_id(), 0.0, search_node_data);
        let next_nodes = discover(&vec![search_node], &mut args);

        assert_eq!(next_nodes.len(), 1);
        assert!(next_nodes[0].score > 0.0);
        assert_eq!(next_nodes[0].data.graph, quick_graph(&[
            "crm:E12_Production1--crm:P14_carried_out_by--crm:E39_Actor1",
            "crm:E12_Production1--karma:classLink--productionURI::d",
            "crm:E39_Actor1--karma:classLink--artistURI::d",
            "crm:E22_Man-Made_Object1--karma:classLink--objectURI::d",
            "crm:E22_Man-Made_Object1--crm:P108i_was_produced_by--crm:E12_Production1",
        ]).0);
    }

    #[test]
    #[ignore]
    /// This test is to enforce repeated called produce
    pub fn test_discover_next_nodes_stable() {
        let input = RustInput::get_input("resources/assembling/museum-jws-crm-s01-s14-rust-input.json");
        let annotator = input.get_annotator();
        let int_graph = IntGraph::new(&input.get_train_sms());

        let sm = input.get_train_sms()[0];
        let mut args = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);
        let seeds = get_started_eliminated_2nodes(&annotator, sm);
        let started_nodes = convert_to_constraint_nodes(sm, &args.int_graph, seeds);

        let next_nodes = discover(&started_nodes, &mut args);
        let graphs = next_nodes.into_iter().map(|n| n.remove_graph()).collect::<Vec<_>>();
        let gold_file = format!("resources/assembling/searching/discovery_constraint_space/{}.json", sm.id);

//        serialize_json(&graphs, &gold_file);
        let gold_graphs: Vec<Graph> = deserialize_json(&gold_file);
        assert_eq!(graphs, gold_graphs);
    }
}