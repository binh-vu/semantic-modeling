use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Serialize, Deserialize, Eq, PartialEq, Hash, Clone)]
pub enum PredicateType {
   OwlDataProp,
   OwlObjectProp,
   OwlAnnotationProp,
   RdfProp
}

#[derive(Serialize, Deserialize)]
pub struct Predicate {
    pub uri: String,
    domains: HashSet<String>,
    ranges: HashSet<String>,
    rdf_type: PredicateType,
    is_rdf_type_reliable: bool,
}

#[derive(Serialize, Deserialize)]
pub struct OntGraphNode {
    uri: String,
    parents_uris: HashSet<String>,
    children_uris: HashSet<String>
}

#[derive(Serialize, Deserialize)]
pub struct OntGraph {
    predicates: Vec<Predicate>,
    class_uris: HashMap<String, OntGraphNode>,
}

impl OntGraph {
    pub fn get_potential_class_node_uris(&self) -> Vec<String> {
        self.class_uris.keys().cloned().collect()
    }

    pub fn get_possible_predicates<'a>(&'a self, source_uri: &str, target_uri: &str) -> IterPredicate<'a> {
        let mut possible_predicates = Vec::new();
        let source = &self.class_uris[source_uri];
        let target = &self.class_uris[target_uri];

        for (i, predicate) in self.predicates.iter().enumerate() {
            if self.is_linkable(predicate, source, target) {
                possible_predicates.push(i);
            }
        }
        
        IterPredicate::new(possible_predicates, self)
    }

    pub fn is_linkable(&self, predicate: &Predicate, source: &OntGraphNode, target: &OntGraphNode) -> bool {
        // can improve this to eliminate all the if-s to just one expression
        // but it will be less readable.
        if predicate.domains.len() > 0 {
            if !predicate.domains.contains(&source.uri) && predicate.domains.intersection(&source.parents_uris).count() == 0 {
                return false;
            }
        }

        if predicate.ranges.len() > 0 {
            if !predicate.ranges.contains(&target.uri) && predicate.ranges.intersection(&target.parents_uris).count() == 0 {
                return false;
            }
        }

        if predicate.is_rdf_type_reliable && (
            predicate.rdf_type == PredicateType::OwlDataProp ||
            predicate.rdf_type == PredicateType::OwlAnnotationProp
        ) {
            // we normally check if this is target.type & rdf_type should be compatible
            // however, since this ont_graph are used to get predicates of class nodes only
            // we are limited to data & annotation properties
            return false;
        }

        return true;
    }
}

pub struct IterPredicate<'a> {
    current_idx: usize,
    pub(super) predicates: Vec<usize>,
    pub(super) graph: &'a OntGraph
}

impl<'a> IterPredicate<'a> {
    pub fn new(predicates: Vec<usize>, graph: &'a OntGraph) -> IterPredicate<'a> {
        IterPredicate {
            current_idx: 0,
            predicates,
            graph
        }
    }
}

impl<'a> Iterator for IterPredicate<'a> {
    type Item = &'a Predicate;

    fn next(&mut self) -> Option<Self::Item> {
        let result = if self.current_idx < self.predicates.len() {
            Some(&self.graph.predicates[self.predicates[self.current_idx]])
        } else {
            None
        };

        self.current_idx += 1;
        result
    }
}

