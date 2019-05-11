use evaluation_metrics::semantic_modeling::alignment::DataNodeMode;
use algorithm::data_structure::graph::*;
use std::collections::HashSet;

use std::iter::FromIterator;
use std::collections::HashMap;
use fnv::FnvHashSet;
use fnv::FnvHashMap;
use std::cmp;
use permutohedron::factorial;

pub const DEFAULT_IGNORE_LABEL_DATA_NODE_ID: i32 = -999_999;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LabelGroup<'a> {
    pub nodes: Vec<&'a Node>,
    pub graph: &'a Graph,
    pub data_node_mode: DataNodeMode
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PairLabelGroup<'a> {
    pub x: LabelGroup<'a>,
    pub x_prime: LabelGroup<'a>
}

#[derive(Debug)]
pub struct StructureGroup<'a> {
    pub nodes: Vec<&'a Node>
}

impl<'a> StructureGroup<'a> {
    #[inline]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }
}

#[derive(Debug, Deserialize, Serialize, Eq, PartialEq, Clone)]
pub struct Bijection {
    pub prime2x: Vec<i32>,
    pub x2prime: Vec<i32>,
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Triple<'a> {
    pub source_id: i32,
    pub target_id: i32,
    pub predicate: &'a str
}

pub type TripleSet<'a> = HashSet<Triple<'a>>;

impl<'a> LabelGroup<'a> {

    pub fn new(graph: &'a Graph, nodes: Vec<&'a Node>, data_node_mode: DataNodeMode) -> LabelGroup<'a> {
        LabelGroup { graph, nodes, data_node_mode }
    }

    pub fn push(&mut self, node: &'a Node) {
        self.nodes.push(node);
    }

    pub fn get_triples(&self) -> HashSet<Triple<'a>> {
        match self.data_node_mode {
            DataNodeMode::NoTouch => {
                self.nodes.iter().flat_map(|n| {
                    let mut res = Vec::new();
                    for e in n.iter_incoming_edges(self.graph).chain(n.iter_outgoing_edges(self.graph)) {
                        res.push(Triple { source_id: e.source_id as i32, target_id: e.target_id as i32, predicate: &e.label });
                    }
                    res
                }).collect::<HashSet<_>>()
            },
            DataNodeMode::IgnoreDataNode => {
                self.nodes.iter().flat_map(|n| {
                    let mut res = Vec::new();
                    for e in n.iter_incoming_edges(self.graph).chain(n.iter_outgoing_edges(self.graph)) {
                        if e.get_target_node(self.graph).is_data_node() {
                            continue;
                        }
                        res.push(Triple { source_id: e.source_id as i32, target_id: e.target_id as i32, predicate: &e.label });
                    }
                    res
                }).collect::<HashSet<_>>()
            },
            DataNodeMode::IgnoreLabelDataNode => {
                self.nodes.iter().flat_map(|n| {
                    let mut res = Vec::new();
                    for e in n.iter_incoming_edges(self.graph).chain(n.iter_outgoing_edges(self.graph)) {
                        let target_id = if e.get_target_node(self.graph).is_data_node() {
                            // so all data target will map to each others
                            DEFAULT_IGNORE_LABEL_DATA_NODE_ID
                        } else {
                            e.target_id as i32
                        };

                        res.push(Triple { source_id: e.source_id as i32, target_id: target_id, predicate: &e.label });
                    }
                    res
                }).collect::<HashSet<_>>()
            }
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// A structure of a node is defined by its links, or we can treat it as a set of triple.
    /// Unbounded nodes should be assumed to be different, therefore a node have unbounded nodes will have
    /// it own structure group.
    ///
    /// We need not consider triple that are impossible to map to node in pred_group. This trick will improve
    /// the performance.
    pub fn group_by_structures(&self, _pred_group: &LabelGroup<'a>) -> Vec<StructureGroup<'a>> {
        self.nodes.iter().map(|&n| StructureGroup { nodes: vec![n] }).collect()
    }
}

impl<'a: 'a2, 'a2> PairLabelGroup<'a> {
    pub fn new(x: LabelGroup<'a>, x_prime: LabelGroup<'a>) -> PairLabelGroup<'a> {
        PairLabelGroup { x, x_prime }
    }

    /// warning: this function is unsafe, and it should only be called from dependent groups function
    /// because we always guarantee that all label groups have at least one element
    pub fn label(&self) -> &'a2 String {
        &self.x.nodes[0].label
    }
}

impl Bijection {
    /// Create an empty bijection, only use to workaround uninitialized error
    pub fn empty() -> Bijection {
        return Bijection {
            x2prime: Vec::new(),
            prime2x: Vec::new(),
        }
    }

    pub fn new(n_x: usize, n_x_prime: usize) -> Bijection {
        Bijection {
            prime2x: vec![-1; n_x_prime],
            x2prime: vec![-1; n_x],
        }
    }

    pub fn new_like(bijection: &Bijection) -> Bijection {
        Bijection {
            prime2x: vec![-1; bijection.prime2x.len()],
            x2prime: vec![-1; bijection.x2prime.len()],
        }
    }

    pub fn clear(&mut self) {
        for i in 0..self.prime2x.len() {
            self.prime2x[i] = -1;
        }
        for i in 0..self.x2prime.len() {
            self.x2prime[i] = -1;
        }
    }

    pub fn append_x(&mut self, x: usize, x_prime: usize) {
        debug_assert_eq!(x, self.x2prime.len());
        self.prime2x[x_prime] = x as i32;
        self.x2prime.push(x_prime as i32);
    }

    pub fn push(&mut self, x: Option<usize>, x_prime: Option<usize>) {
        if x_prime.is_some() {
            self.prime2x[x_prime.unwrap()] = match x {
                None => -1,
                Some(v) => v as i32
            }
        }

        if x.is_some() {
            self.x2prime[x.unwrap()] = match x_prime {
                None => -1,
                Some(v) => v as i32
            }
        }
    }

    pub fn push_x_prime(&mut self, x: Option<usize>, x_prime: usize) {
        match x {
            None => {
                self.prime2x[x_prime] = -1;
            },
            Some(v) => {
                self.prime2x[x_prime] = v as i32;
                self.x2prime[v] = x_prime as i32;
            }
        }
    }

    pub fn push_x(&mut self, x: usize, x_prime: Option<usize>) {
        match x_prime {
            None => {
                self.x2prime[x] = -1;
            },
            Some(v) => {
                self.x2prime[x] = v as i32;
                self.prime2x[v] = x as i32;
            }
        }
    }

    pub fn push_both(&mut self, x: usize, x_prime: usize) {
        self.x2prime[x] = x_prime as i32;
        self.prime2x[x_prime] = x as i32;
    }

    pub fn extends(&self, bijection: &Bijection) -> Bijection {
        let mut new_bijection = Bijection {
            prime2x: self.prime2x.clone(),
            x2prime: self.x2prime.clone(),
        };

        for (x_prime, &x) in bijection.prime2x.iter().enumerate() {
            if x != -1 {
                new_bijection.prime2x[x_prime] = x;
            }
        }

        for (x, &x_prime) in bijection.x2prime.iter().enumerate() {
            if x_prime != -1 {
                new_bijection.x2prime[x] = x_prime;
            }
        }

        new_bijection
    }

    pub fn extends_(&mut self, bijection: &Bijection) {
        for (x_prime, &x) in bijection.prime2x.iter().enumerate() {
            if x != -1 {
                self.prime2x[x_prime] = x;
            }
        }

        for (x, &x_prime) in bijection.x2prime.iter().enumerate() {
            if x_prime != -1 {
                self.x2prime[x] = x_prime;
            }
        }
    }

    #[inline]
    pub fn is_gold_node_bounded(&self, node_id: usize) -> bool {
        self.x2prime[node_id] != -1
    }

    #[inline]
    pub fn is_pred_node_bounded(&self, node_id: usize) -> bool {
        self.prime2x[node_id] != -1
    }

    pub fn to_x_prime<'a>(&self, x: usize) -> i32 {
        self.x2prime[x] 
    }

    pub fn to_x<'a>(&self, x_prime: usize) -> i32 {
        self.prime2x[x_prime]
    }

    #[inline]
    pub fn has_x_prime(&self, x: usize) -> bool {
        self.x2prime[x] != -1
    }

    #[inline]
    pub fn has_x(&self, x_prime: usize) -> bool {
        self.prime2x[x_prime] != -1
    }
}

