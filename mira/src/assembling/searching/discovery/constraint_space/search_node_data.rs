use fnv::FnvHashMap;
use algorithm::prelude::*;
use im::OrdSet as IOrdSet;
use assembling::searching::beam_search::*;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Bijection {
    prime2x: Vec<usize>,
    x2prime: FnvHashMap<usize, usize>
}

#[derive(Clone, Debug)]
pub struct IntTreeSearchNodeData {
    pub graph: Graph,
    pub(super) remained_terminals: IOrdSet<String>,
    pub(super) bijections: Vec<Bijection>
}

impl Bijection {
    pub fn new(prime2x: Vec<usize>, x2prime: FnvHashMap<usize, usize>) -> Bijection {
        Bijection { prime2x, x2prime }
    }

    pub fn from_pairs(prime2x: Vec<usize>) -> Bijection {
        let x2prime = prime2x.iter().enumerate()
            .map(|(x_prime, &x)| (x, x_prime))
            .collect::<FnvHashMap<usize, usize>>();

        Bijection {
            prime2x,
            x2prime
        }
    }

    pub fn append(&mut self, x_id: usize, x_prime_id: usize) {
        debug_assert_eq!(x_prime_id, self.prime2x.len());
        self.prime2x.push(x_id);
        self.x2prime.insert(x_id, x_prime_id);
    }

    #[inline]
    pub fn to_x(&self, x_prime_id: usize) -> usize {
        self.prime2x[x_prime_id]
    }

    #[inline]
    pub fn to_x_prime(&self, x_id: usize) -> Option<&usize> {
        self.x2prime.get(&x_id)
    }
}

impl IntTreeSearchNodeData {
    pub fn new(graph: Graph, remained_terminals: IOrdSet<String>, bijections: Vec<Bijection>) -> IntTreeSearchNodeData {
        IntTreeSearchNodeData { graph, remained_terminals, bijections }
    }

    pub fn get_id(&self) -> String {
        get_acyclic_consistent_unique_hashing(&self.graph)
    }

    pub fn update_graph(&mut self, g: Graph) {
        self.graph = g;
    }

    pub fn update_bijections(&mut self, bijections: Vec<Bijection>) {
        self.bijections = bijections;
    }
}

impl SearchNodeExtraData for IntTreeSearchNodeData {
    fn is_terminal(&self) -> bool {
        self.remained_terminals.len() == 0
    }

    fn get_graph(&self) -> &Graph {
        &self.graph
    }

    fn remove_graph(self) -> Graph {
        self.graph
    }
}