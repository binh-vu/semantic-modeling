use std::cmp::Ordering;
use std::collections::HashMap;
use algorithm::data_structure::graph::Graph;


pub trait SearchNodeExtraData: Clone {
    fn is_terminal(&self) -> bool;
    fn get_graph(&self) -> &Graph;
    fn remove_graph(self) -> Graph;
}


#[derive(Debug, Clone)]
pub struct SearchNode<T: SearchNodeExtraData> {
    pub id: String,
    pub score: f64,
    pub data: T
}

impl<T: SearchNodeExtraData> SearchNode<T> {
    pub fn new(id: String, score: f64, data: T) -> SearchNode<T> {
        SearchNode {
            id,
            score,
            data
        }    
    }

    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.data.is_terminal()
    }

    #[inline]
    pub fn get_graph(&self) -> &Graph {
        self.data.get_graph()
    }

    #[inline]
    pub fn remove_graph(self) -> Graph {
        self.data.remove_graph()
    }
}

pub type ExploringNodes<'a, T> = Vec<SearchNode<T>>;
pub type DiscoverFunc<'a, T, A> = &'a Fn(&ExploringNodes<T>, &mut A) -> Vec<SearchNode<T>>;
pub type EarlyStopFunc<'a, T, A> = &'a Fn(&mut SearchArgs<'a, T, A>, usize, &SearchNode<T>) -> bool;
pub type CompareSearchNodeFunc<'a, T> = &'a Fn(&SearchNode<T>, &SearchNode<T>) -> Ordering;

pub struct SearchArgs<'a, T: 'a + SearchNodeExtraData, A: 'a> {
    pub sm_id: &'a str,
    pub beam_width: usize,
    pub n_results: usize,
    pub enable_tracking: bool,
    pub tracker: Vec<Vec<SearchNode<T>>>,
    pub discover: DiscoverFunc<'a, T, A>,
    pub should_stop: EarlyStopFunc<'a, T, A>,
    pub compare_search_node: CompareSearchNodeFunc<'a, T>,
    pub extra_args: A
}

impl<'a, T: 'a + SearchNodeExtraData, A: 'a> SearchArgs<'a, T, A> {
    pub fn log_search_nodes(&mut self, _iter_no: usize, search_nodes: &ExploringNodes<T>) {
        if self.enable_tracking {
            let mut search_nodes = search_nodes.iter().cloned().collect::<Vec<_>>();
            search_nodes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            self.tracker.push(search_nodes);
        }
    }

}

pub fn default_compare_search_node<T: SearchNodeExtraData>(a: &SearchNode<T>, b: &SearchNode<T>) -> Ordering {
    // higher is better
    b.score.partial_cmp(&a.score).unwrap()
}
