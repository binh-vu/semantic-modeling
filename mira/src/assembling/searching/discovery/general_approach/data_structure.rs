use std::rc::Rc;
use std::collections::HashMap;
use im::HashSet as IHashSet;
use algorithm::data_structure::graph::*;
use assembling::searching::beam_search::*;
use rdb2rdf::models::semantic_model::*;
use assembling::searching::general_approach::graph_explorer::GraphExplorer;
use assembling::searching::general_approach::graph_explorer_builder::GraphExplorerBuilder;
use assembling::searching::discovery::general_approach::merge_graph::MergeGraph;
use assembling::searching::discovery::general_approach::merge_plan::make_merge_plans4case1;
use assembling::searching::discovery::general_approach::merge_plan::make_merge_plans;
use assembling::searching::beam_search::SearchNodeExtraData;

pub type ProbCandidateSMFunc<'a> = &'a Fn(Vec<Graph>) -> Vec<(Graph, f64)>;
pub type PreSMFilterFunc<'a> = &'a Fn(&Graph) -> bool;

pub struct GeneralDiscoveryArgs<'a> {
    pub beam_width: usize,
    pub prob_candidate_sms: ProbCandidateSMFunc<'a>,
    pub pre_sm_filter: PreSMFilterFunc<'a>,
    pub graph_explorer_builder: GraphExplorerBuilder<'a>,
}

pub type GeneralDiscoveryNode = SearchNode<GraphDiscovery>;

#[derive(Debug, Clone)]
pub struct GraphDiscovery {
    pub(super) terminals_to_index: Rc<HashMap<String, usize>>,
    pub(super) g_explorers: Rc<Vec<GraphExplorer>>,
    pub(super) g_terminals: Rc<Vec<Graph>>,
    pub(super) remained_terminals: IHashSet<String>,

    pub current_graph: Graph,
    pub current_graph_explorer: GraphExplorer,
    pub current_score: f64,
}

impl SearchNodeExtraData for GraphDiscovery {
    fn is_terminal(&self) -> bool {
        self.remained_terminals.len() == 0
    }

    fn get_graph(&self) -> &Graph {
        &self.current_graph
    }

    fn remove_graph(self) -> Graph {
        self.current_graph
    }
}

impl GraphDiscovery {
    pub fn init(attributes: &[Attribute], args: &mut GeneralDiscoveryArgs) -> GraphDiscovery {
        let mut terminals_to_index: HashMap<String, usize> = Default::default();
        let mut g_explorers: Vec<GraphExplorer> = Default::default();
        let mut g_terminals: Vec<Graph> = Default::default();
        let mut remained_terminals: IHashSet<String> = Default::default();

        for (i, attr) in attributes.iter().enumerate() {
            let mut g = Graph::new("".to_owned(), true, true, false);
            g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));

            terminals_to_index.insert(attr.label.clone(), i);
            g_explorers.push(args.graph_explorer_builder.build(&g));
            g_terminals.push(g);

            remained_terminals.insert(attr.label.clone());
        }

        let current_graph = Graph::new("".to_owned(), true, true, false);

        GraphDiscovery {
            current_graph_explorer: GraphExplorer::new(&current_graph),
            current_graph,
            current_score: 0.0,
            terminals_to_index: Rc::new(terminals_to_index),
            g_explorers: Rc::new(g_explorers),
            g_terminals: Rc::new(g_terminals),
            remained_terminals,
        }
    }

    pub fn merge_terminal(&self, terminal: &str, g_explorer: GraphExplorer, g: Graph, score: f64) -> GraphDiscovery {
        GraphDiscovery {
            current_graph: g,
            current_graph_explorer: g_explorer,
            current_score: score,
            terminals_to_index: Rc::clone(&self.terminals_to_index),
            g_explorers: Rc::clone(&self.g_explorers),
            g_terminals: Rc::clone(&self.g_terminals),
            remained_terminals: self.remained_terminals.without(terminal),
        }
    }

    pub fn replace_current_state(&self, remained_terminals: IHashSet<String>, g_explorer: GraphExplorer, g: Graph, score: f64) -> GraphDiscovery {
        GraphDiscovery {
            current_graph: g,
            current_graph_explorer: g_explorer,
            current_score: score,
            terminals_to_index: Rc::clone(&self.terminals_to_index),
            g_explorers: Rc::clone(&self.g_explorers),
            g_terminals: Rc::clone(&self.g_terminals),
            remained_terminals
        }
    }

    #[inline]
    pub fn get_current_id(&self) -> String {
        get_acyclic_consistent_unique_hashing(&self.current_graph)
    }

    #[inline]
    pub fn make_merge_plans4case1(&self, terminal: &str) -> Vec<MergeGraph> {
        let idx = self.terminals_to_index[terminal];
        return make_merge_plans4case1(
            &self.current_graph, &self.g_terminals[idx],
            &self.current_graph_explorer, &self.g_explorers[idx])
            .into_iter()
            .map(|plan| {
                MergeGraph::new(
                    &self.current_graph, &self.g_terminals[idx],
                    plan.int_tree, plan.int_a, plan.int_b,
                )
            }).collect();
    }

    #[inline]
    pub fn make_merge_plans(&self, terminal: &str) -> Vec<MergeGraph> {
        let idx = self.terminals_to_index[terminal];
        return make_merge_plans(
            &self.current_graph, &self.g_terminals[idx],
            &self.current_graph_explorer, &self.g_explorers[idx])
            .into_iter()
            .map(|plan| MergeGraph::new(
                &self.current_graph, &self.g_terminals[idx],
                plan.int_tree, plan.int_a, plan.int_b,
            )).collect();
    }
}