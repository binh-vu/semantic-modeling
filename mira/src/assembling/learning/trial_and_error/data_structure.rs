use assembling::searching::beam_search::*;
use algorithm::data_structure::graph::*;
use im::OrdSet as IOrdSet;
use im::HashMap as IHashMap;
use assembling::features::statistic::Statistic;
use std::collections::HashMap;
use rdb2rdf::models::semantic_model::SemanticType;
use rdb2rdf::models::semantic_model::SemanticModel;
use assembling::models::annotator::Annotator;
use std::ops::Deref;
use std::rc::Rc;
use std::collections::HashSet;
use utils::dict_has;
use utils::dict_get;
use std::collections::hash_set;
use fnv::FnvHashMap;
use std::cmp::Ordering;

#[derive(Clone, Debug)]
pub struct Mount {
    pub(super) class_id: usize,
    pub(super) pred_idx: usize,
    pub(super) is_done: bool
}

impl Mount {
    pub fn new(class_id: usize, pred_idx: usize, is_done: bool) -> Mount {
        Mount {
            class_id, pred_idx, is_done
        }
    }

    pub fn unroll<'a>(&self, graph: &'a Graph, args: &'a TrialErrorSearchArgs) -> (&'a Node, &'a String) {
        let mnt_node = graph.get_node_by_id(self.class_id);
        (
            mnt_node,
            &args.trial_error_exploring.sorted_l_given_s(&mnt_node.label)[self.pred_idx]
        )
    }

    pub fn is_class_mount(&self, graph: &Graph, args: &TrialErrorSearchArgs) -> bool {
        // same like outside is_class_mount (convenience method)
        let (mnt_subj, mnt_pred) = self.unroll(graph, args);
        return dict_has(&args.trial_error_exploring.o_given_sl, &mnt_subj.label, mnt_pred);
    }

    pub fn next_mount<'a>(&self, graph: &'a Graph, args: &'a TrialErrorSearchArgs) -> Option<Mount> {
        let mnt_subj = &graph.get_node_by_id(self.class_id).label;

        if args.trial_error_exploring.sorted_l_given_s(mnt_subj).len() == self.pred_idx + 1 {
            // forward to next class node
            let mut mount_subj_id = self.class_id;
            let mut mount_pred_id = 0;
            let mut mount_is_done = false;

            for n in graph.iter_class_nodes() {
                if n.id > mount_subj_id {
                    // next class node always has id greater than current node
                    mount_subj_id = n.id;
                    if n.n_outgoing_edges > 0 {
                        // may be this class node has some nodes before, so we need to generate pred_id correctly
                        let e = graph.get_edge_by_id(n.outgoing_edges[0]);
                        for (i, e_lbl) in args.trial_error_exploring.sorted_l_given_s(&n.label).iter().enumerate() {
                            if e_lbl == &e.label {
                                mount_pred_id = i;
                                mount_is_done = true;
                                break;
                            }
                        }
                    }
                    break;
                }
            }

            if mount_subj_id == self.class_id {
                // no more class node
                None
            } else {
                Some(Mount::new(mount_subj_id, mount_pred_id, mount_is_done))
            }
        } else {
            Some(Mount::new(self.class_id, self.pred_idx + 1, false))
        }
    }
}


#[derive(Clone, Debug)]
pub struct TrialErrorNodeData {
    pub(super) remained_attributes: IOrdSet<String>,
    pub(super) current_graph: Graph,
    pub(super) mount: Option<Mount>,
}

impl SearchNodeExtraData for TrialErrorNodeData {
    fn is_terminal(&self) -> bool {
        self.mount.is_none() || self.remained_attributes.len() == 0
    }

    fn get_graph(&self) -> &Graph {
        &self.current_graph
    }

    fn remove_graph(self) -> Graph {
        self.current_graph
    }
}


impl TrialErrorNodeData {
    pub fn new(remained_attributes: IOrdSet<String>, mount: Option<Mount>, graph: Graph) -> TrialErrorNodeData {
        TrialErrorNodeData {
            remained_attributes,
            current_graph: graph,
            mount
        }
    }

    pub fn into_search_node(self, score: f64) -> SearchNode<TrialErrorNodeData> {
        SearchNode::new(
            get_acyclic_consistent_unique_hashing(&self.current_graph),
            score,
            self
        )
    }
}

pub struct TrialErrorExploring {
    pub(super) sorted_l_given_s: HashMap<String, Vec<String>>,
    pub(super) o_given_sl: HashMap<(String, String), HashSet<String>>,
    pub(super) o_given_sl_ordered: HashMap<(String, String), Vec<String>>, 
    pub(super) attr2idx: HashMap<String, usize>,
}

impl TrialErrorExploring {

    pub fn new(annotator: &Annotator, sm: &SemanticModel) -> TrialErrorExploring {
        let o_given_sl = annotator.statistic.p_o_given_sl.iter()
            .map(|(key, value)| {
                let mut list = value.keys().cloned().collect::<HashSet<_>>();
                (key.clone(), list)
            })
            .collect::<HashMap<_, _>>();

        let o_given_sl_ordered = annotator.statistic.p_o_given_sl.iter()
            .map(|(key, value)| {
                let mut list = value.keys().cloned().collect::<Vec<_>>();
                list.sort_unstable();
                (key.clone(), list)
            })
            .collect::<HashMap<_, _>>();

        let sorted_l_given_s = annotator.statistic.p_l_given_s.iter()
            .map(|(key, value)| {
                let mut list: Vec<String> = value.keys().cloned().collect::<Vec<_>>();
                // do this sort first to make sure it always in order
                // so the result is consistent
                list.sort_unstable();

                let pk = annotator.primary_key.get_primary_key(key);

                list.sort_by(|a, b| {
                    if a == pk {
                        return Ordering::Less;
                    }

                    if b == pk {
                        return Ordering::Greater;
                    }

                    if value[b] == value[a] {
                        let a_is_class = dict_has(&o_given_sl, key, a);
                        let b_is_class = dict_has(&o_given_sl, key, b);
                        
                        if a_is_class && !b_is_class {
                            Ordering::Greater
                        } else if !a_is_class && b_is_class {
                            Ordering::Less
                        } else {
                            Ordering::Equal
                        }
                    } else {
                        value[b].partial_cmp(&value[a]).unwrap()
                    }
                });


                (key.clone(), list)
            })
            .collect::<HashMap<_, _>>();

        let attr2idx = sm.attrs.iter()
            .enumerate()
            .map(|(i, attr)| (attr.label.clone(), i))
            .collect::<HashMap<_, _>>();

        // need to make sure that the first important predicate always its primary key
        // for (s, sorted_l) in sorted_l_given_s.iter() {
            // assert_eq!(annotator.primary_key.get_primary_key(s), sorted_l[0]);
            // === [DEBUG] DEBUG CODE START HERE ===
            // println!("[DEBUG] s = {}, sorted_l: {:?}", s, sorted_l);
            // === [DEBUG] DEBUG CODE END   HERE ===
        // }
        // println!("[DEBUG] sorted_l_given_s = {:#?}", sorted_l_given_s);

        TrialErrorExploring {
            sorted_l_given_s,
            o_given_sl,
            o_given_sl_ordered,
            attr2idx
        }
    }

    pub fn sorted_l_given_s(&self, s: &str) -> &[String] {
        &self.sorted_l_given_s[s]
    }
}

pub type ProbCandidateSMFunc<'a> = &'a Fn(Vec<Graph>) -> Vec<(Graph, f64)>;

pub struct TrialErrorSearchArgs<'a> {
    pub beam_width: usize,
    pub prob_candidate_sms: ProbCandidateSMFunc<'a>,
    pub trial_error_exploring: TrialErrorExploring,
    pub max_n_duplications: usize,
    pub sm: &'a SemanticModel
}

impl<'a> TrialErrorSearchArgs<'a> {

    #[inline]
    pub fn get_attr_stypes(&self, attr: &str) -> &[SemanticType]{
        &self.sm.attrs[self.trial_error_exploring.attr2idx[attr]].semantic_types
    }
}

pub type TrialErrorSearchNode = SearchNode<TrialErrorNodeData>;