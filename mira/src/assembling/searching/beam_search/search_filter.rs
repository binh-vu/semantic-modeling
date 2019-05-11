use settings::Settings;
use algorithm::prelude::*;

pub struct SearchFilter {
    max_n_duplications: usize,
    max_n_duplication_types: usize,
    allow_one_empty_data_hope: bool
}

impl SearchFilter {
    pub fn new(settings: &Settings, allow_one_empty_data_hope: bool) -> SearchFilter {
        SearchFilter {
            max_n_duplications: settings.mrf.max_n_duplications,
            max_n_duplication_types: settings.mrf.max_n_duplication_types,
            allow_one_empty_data_hope
        }
    }

    /// Filter unlikely graph
    pub fn filter(&self, g: &Graph) -> bool {
        for n in g.iter_class_nodes() {
            // FILTER 2 hop middle nodes
            if n.n_incoming_edges == 1 && n.n_outgoing_edges == 1 {
                let child_node = n.iter_children(&g).next().unwrap();
                if child_node.is_class_node() {
                    if self.allow_one_empty_data_hope {
                        if child_node.n_outgoing_edges == 1 && child_node.iter_children(&g).next().unwrap().is_class_node() {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }

            let mut n_duplication_types = 0;
            for group in group_by(n.iter_outgoing_edges(&g), |e| &e.label) {
                if group.len() >= self.max_n_duplications {
                    return false;
                }

                if group.len() > 1 {
                    n_duplication_types += 1;
                }
            }

            if n_duplication_types >= self.max_n_duplication_types {
                return false;
            }
        }

        return true;
    }
}