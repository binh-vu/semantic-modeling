use assembling::features::primary_key::PrimaryKey;
use itertools::Itertools;
use std::collections::HashMap;
use rdb2rdf::models::semantic_model::SemanticModel;
use std::ops::AddAssign;
use std::cmp::Ordering;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeStructureSpace {
    // parent_idx and child_idx always different, parent_idx < child_idx
    pub label: String,
    pub parents: HashMap<(String, String), usize>,
    pub children: HashMap<(String, String), usize>,
    pub child_pred2idx: HashMap<String, usize>,
    is_class_child: Vec<bool>,
    child_usage_count: Vec<usize>
}

impl NodeStructureSpace {
    pub fn get_parent_idx(&self, link_lbl: &str, parent_lbl: &str) -> Option<&usize> {
        self.parents.get(&(link_lbl.to_owned(), parent_lbl.to_owned()))
    }

    pub fn get_child_idx(&self, link_lbl: &str, child_lbl: &str) -> Option<&usize> {
        self.children.get(&(link_lbl.to_owned(), child_lbl.to_owned()))
    }

    pub fn has_smaller_index(&self, link_lbl_a: &str, link_lbl_b: &str) -> bool {
        self.child_pred2idx[link_lbl_a] < self.child_pred2idx[link_lbl_b]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LocalStructure {
    pub node_structure_space: HashMap<String, NodeStructureSpace>
}

#[derive(Debug)]
struct RawNodeStruct<'a> {
    parents: HashMap<(&'a str, &'a str), usize>,
    children: HashMap<(&'a str, &'a str), usize>,
}

impl Default for LocalStructure {
    fn default() -> Self {
        LocalStructure {
            node_structure_space: Default::default()
        }
    }
}

impl LocalStructure {
    pub fn new(train_sms: &[&SemanticModel], primary_key: &PrimaryKey) -> LocalStructure {
        let mut node_structure_space: HashMap<String, NodeStructureSpace> = Default::default();

        // make raw structure space first
        let mut raw_node_structure_space: HashMap<&str, RawNodeStruct> = Default::default();
        for sm in train_sms {
            for n in sm.graph.iter_class_nodes() {
                if !raw_node_structure_space.contains_key(n.label.as_str()) {
                    raw_node_structure_space.insert(&n.label, RawNodeStruct {
                        parents: Default::default(),
                        children: Default::default(),
                    });
                }

                let p_edge = n.first_incoming_edge(&sm.graph);
                if let Some(p_edge) = p_edge {
                    let triple = (p_edge.label.as_str(), p_edge.get_source_node(&sm.graph).label.as_str());
                    raw_node_structure_space.get_mut(n.label.as_str()).unwrap()
                        .parents.entry(triple).or_insert(0).add_assign(1);
                }

                for e in n.iter_outgoing_edges(&sm.graph).unique_by(|e| &e.label) {
                    let target = e.get_target_node(&sm.graph);
                    let triple = (e.label.as_str(), if target.is_data_node() {
                        "DATA_NODE"
                    } else {
                        &target.label
                    });

                    raw_node_structure_space.get_mut(n.label.as_str()).unwrap()
                        .children.entry(triple).or_insert(0).add_assign(1);
                }
            }
        }

        // make node structure space
        for (&n_lbl, space) in raw_node_structure_space.iter_mut() {
            let pk = if primary_key.contains(n_lbl) {
                primary_key.get_primary_key(n_lbl)
            } else {
                "__no_pk__"
            };

            let mut children_attrs = space.children.keys().collect::<Vec<_>>();
            children_attrs.sort_by(|a, b| {
                // always make primary key the first element!!
                if a.0 == pk {
                    Ordering::Less
                } else if b.0 == pk {
                    Ordering::Greater
                } else if space.children[b] == space.children[a] && a.0 != b.0 {
                    // compare by lexical order
                    b.0.cmp(&a.0)
                } else {
                    space.children[b].cmp(&space.children[a])
                }
            });

            let child_pred2idx = children_attrs.iter()
                .unique_by(|a| a.0)
                .enumerate()
                .map(|(i, a)| (a.0.to_owned(), i))
                .collect::<HashMap<_, _>>();

            node_structure_space.insert(n_lbl.to_owned(), NodeStructureSpace {
                label: n_lbl.to_owned(),
                parents: space.parents.keys().enumerate()
                    .map(|(i, v)| ((v.0.to_owned(), v.1.to_owned()), i))
                    .collect(),
                children: children_attrs.iter().enumerate()
                    .map(|(i, v)| ((v.0.to_owned(), v.1.to_owned()), i + space.parents.len()))
                    .collect(),
                child_pred2idx,
                is_class_child: children_attrs.iter().map(|v| v.1 != "DATA_NODE").collect(),
                child_usage_count: children_attrs.iter().map(|v| space.children[v]).collect()
            });
        }

        LocalStructure {
            node_structure_space
        }
    }
}