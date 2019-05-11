use rdb2rdf::prelude::*;
use algorithm::prelude::*;
use assembling::searching::banks::*;
use super::*;


#[derive(PartialEq, Eq, Debug, Clone, Deserialize, Serialize)]
pub enum MergePlanType {
    // don't need to create any extra nodes, just attach the mount to the graph
    Attach,
    MergeIntoTree,
    MergeIntoMount,
    MergeTriangle
}

#[derive(PartialEq, Eq, Debug, Deserialize, Serialize)]
pub struct MergePlan {
    plan_type: MergePlanType,
    mnt: usize,
    first_edges: Vec<usize>,
    second_edges: Vec<usize>
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct STypeMount<'a> {
    pub(super) edge: &'a IntEdge
}


impl MergePlan {
    fn attach(mnt: &STypeMount) -> MergePlan {
        MergePlan {
            plan_type: MergePlanType::Attach,
            mnt: mnt.edge.id,
            first_edges: Vec::new(),
            second_edges: Vec::new(),
        }
    }

    /// Create a merge plan from set of edges where first edge is connected to int_sub_graph
    /// last edge end at class_node of mnt
    fn merge_into_tree(mnt: &STypeMount, edges: Vec<usize>) -> MergePlan {
        MergePlan {
            plan_type: MergePlanType::MergeIntoTree,
            mnt: mnt.edge.id,
            first_edges: edges,
            second_edges: Vec::new(),
        }
    }

    /// Create a merge plan from set of edges where first edge is connected to class-node of mnt
    /// last edge end at root of int_sub_graph
    fn merge_into_mount(mnt: &STypeMount, edges: Vec<usize>) -> MergePlan {
        MergePlan {
            plan_type: MergePlanType::MergeIntoMount,
            mnt: mnt.edge.id,
            first_edges: edges,
            second_edges: Vec::new(),
        }
    }

    /// Create a merge plan from two separated path, both start at new root nodes, tree_edges end
    /// at root of tree, and mnt_edges end at class node of mount
    fn merge_triangle(mnt: &STypeMount, tree_edges: Vec<usize>, mnt_edges: Vec<usize>) -> MergePlan {
        MergePlan {
            plan_type: MergePlanType::MergeTriangle,
            mnt: mnt.edge.id,
            first_edges: tree_edges,
            second_edges: mnt_edges
        }
    }

    pub fn proceed(&self, tree: &IntSubGraph, attr: &Attribute, args: &IntTreeSearchArgs) -> (Graph, Bijection) {
        // create new graph by applying this merge plan
        let mut g = tree.graph.clone();
        let mut bijection = tree.bijection.clone();

        match self.plan_type {
            MergePlanType::Attach => {
                let e = args.int_graph.graph.get_edge_by_id(self.mnt);
                let target_prime_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));

                bijection.append(e.target_id, target_prime_id);
                g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));

                return (g, bijection);
            },
            MergePlanType::MergeIntoTree => {
                for &eid in self.first_edges.iter() {
                    let e = args.int_graph.graph.get_edge_by_id(eid);
                    let target = e.get_target_node(&args.int_graph.graph);
                    let target_prime_id = g.add_node(Node::new(target.kind.clone(), target.label.clone()));
                    bijection.append(target.id, target_prime_id);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));
                }

                let e = args.int_graph.graph.get_edge_by_id(self.mnt);
                let target_prime_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));

                bijection.append(e.target_id, target_prime_id);
                g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));

                return (g, bijection);
            },
            MergePlanType::MergeIntoMount => {
                let e = args.int_graph.graph.get_edge_by_id(self.mnt);
                let source = e.get_source_node(&args.int_graph.graph);
                let source_prime_id = g.add_node(Node::new(NodeType::ClassNode, source.label.clone()));
                let target_prime_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));
                bijection.append(e.source_id, source_prime_id);
                bijection.append(e.target_id, target_prime_id);
                g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));

                for i in 0..(self.first_edges.len() - 1) {
                    let e = args.int_graph.graph.get_edge_by_id(self.first_edges[i]);
                    let target = e.get_target_node(&args.int_graph.graph);
                    let target_prime_id = g.add_node(Node::new(target.kind.clone(), target.label.clone()));
                    bijection.append(target.id, target_prime_id);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));
                }

                let e = args.int_graph.graph.get_edge_by_id(*self.first_edges.last().unwrap());
                g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), *bijection.to_x_prime(e.target_id).unwrap()));

                return (g, bijection);
            },
            MergePlanType::MergeTriangle => {
                let e = args.int_graph.graph.get_edge_by_id(self.first_edges[0]);
                let source = e.get_source_node(&args.int_graph.graph);
                let source_prime_id = g.add_node(Node::new(NodeType::ClassNode, source.label.clone()));
                bijection.append(e.source_id, source_prime_id);

                for i in 0..(self.first_edges.len() - 1) {
                    let e = args.int_graph.graph.get_edge_by_id(self.first_edges[i]);
                    let target = e.get_target_node(&args.int_graph.graph);
                    let target_prime_id = g.add_node(Node::new(target.kind.clone(), target.label.clone()));
                    bijection.append(target.id, target_prime_id);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));
                }

                let e = args.int_graph.graph.get_edge_by_id(*self.first_edges.last().unwrap());
                g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), *bijection.to_x_prime(e.target_id).unwrap()));

                for &eid in self.second_edges.iter() {
                    let e = args.int_graph.graph.get_edge_by_id(eid);
                    let target = e.get_target_node(&args.int_graph.graph);
                    let target_prime_id = g.add_node(Node::new(target.kind.clone(), target.label.clone()));
                    bijection.append(target.id, target_prime_id);
                    g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));
                }

                let e = args.int_graph.graph.get_edge_by_id(self.mnt);
                let target_prime_id = g.add_node(Node::new(NodeType::DataNode, attr.label.clone()));

                bijection.append(e.target_id, target_prime_id);
                g.add_edge(Edge::new(EdgeType::Unspecified, e.label.clone(), *bijection.to_x_prime(e.source_id).unwrap(), target_prime_id));

                return (g, bijection);
            }
        }
    }

    pub fn find_merge_plans<'a>(tree: &'a IntSubGraph<'a>, attr: &Attribute, args: &'a IntTreeSearchArgs) -> Vec<MergePlan> {
        let mut merged_plans = Vec::new();

        for stype in &attr.semantic_types {
            for mount in find_all_mounts(stype, &args.int_graph) {
                if let Some(x_prime) = tree.bijection.to_x_prime(mount.edge.source_id) {
                    // it can directly map to tree
                    if tree.bijection.to_x_prime(mount.edge.target_id).is_none() {
                        merged_plans.push(MergePlan::attach(&mount));
                    }
                } else {
                    merged_plans.append(&mut get_merge_plan_into_mount(tree, &mount, args));
                    merged_plans.append(&mut get_merge_plan_into_tree(tree, &mount, args));
                    merged_plans.append(&mut get_merge_plan_triangle(tree, &mount, args));
                }
            }
        }

        merged_plans
    }

    /// Telling a merge plan is good or not based on heuristics
    #[inline]
    pub fn is_good_plan(&self, max_n_empty_hop: usize) -> bool {
        match self.plan_type {
            MergePlanType::Attach => true,
            MergePlanType::MergeIntoTree | MergePlanType::MergeIntoMount => {
                // 2 edges is 1 hop, 3 edges is 2 hops...
                self.first_edges.len() - 1 <= max_n_empty_hop
            },
            MergePlanType::MergeTriangle => {
                // 2 edges is 1 hop, 3 edges is 2 hops... (don't count root node)
                self.first_edges.len() + self.second_edges.len() - 2 <= max_n_empty_hop
            }
        }
    }
}

/// Get a merge plan where stype_mount is merged to tree as a subtree
///
/// The algorithm works by finding a path from any class node of tree to class_node in STypeMount (reversely)
/// Note that class_node in STypeMount cannot be in tree
fn get_merge_plan_into_tree<'a>(tree: &'a IntSubGraph<'a>, stype_mount: &'a STypeMount, args: &IntTreeSearchArgs) -> Vec<MergePlan> {
    let mut plans = Vec::new();

    // loop through each node in tree, and find if there is a path between it to stype mount
    for n in tree.graph.iter_nodes() {
        let paths = args.path_index.get_path_between(tree.bijection.to_x(n.id), stype_mount.edge.source_id);
        for path in paths {
            // if path[0] is an edge in n outgoing edges, we should ignore it
            // because it pass through tree
            let child_id = tree.bijection.to_x_prime(tree.int_graph.graph.get_edge_by_id(path[0]).target_id);
            if child_id.is_some() && tree.children_index.has_child(n.id, *child_id.unwrap()) {
                continue;
            }

            plans.push(MergePlan::merge_into_tree(stype_mount, path.clone()));
        }
    }

    return plans;
}

/// Get a merge plan where root of tree is merged to stype_mount
///
/// The algorithm works by finding a path from root to class_node in STypeMount
fn get_merge_plan_into_mount<'a>(tree: &'a IntSubGraph<'a>, stype_mount: &'a STypeMount, args: &IntTreeSearchArgs) -> Vec<MergePlan> {
    let mut plans = Vec::new();
    let root_id = tree.bijection.to_x(tree.graph.get_first_root_node().unwrap().id);
    for path in args.path_index.get_path_between(stype_mount.edge.source_id, root_id) {
        plans.push(MergePlan::merge_into_mount(stype_mount, path.clone()));
    }

    plans
}

/// Get a merge plan where we have to find a common node, and a list of edges to connect root of tree & class_node
/// of STypeMount to this common nodes
///
/// The algorithm works by find a set of common nodes, and find a path from this node to root of tree and class node
fn get_merge_plan_triangle<'a>(tree: &'a IntSubGraph<'a>, stype_mount: &'a STypeMount, args: &IntTreeSearchArgs) -> Vec<MergePlan> {
    let mut plans = Vec::new();
    let root_id = tree.bijection.to_x(tree.graph.get_first_root_node().unwrap().id);
    let mount_ancestor_ids = args.ancestor_index.get_ancestors(stype_mount.edge.source_id);

    for &nid in args.ancestor_index.iter_ancestors(root_id) {
        if mount_ancestor_ids.contains(&nid) {
            // common node
            for mount_path in args.path_index.get_path_between(nid, stype_mount.edge.source_id) {
                for root_path in args.path_index.get_path_between(nid, root_id) {
                    if mount_path[0] != root_path[0] {
                        plans.push(MergePlan::merge_triangle(stype_mount, root_path.clone(), mount_path.clone()));
                    }
                }
            }
        }
    }

    plans
}

/// Find all mounts that an stype can be mount in int_graph
pub fn find_all_mounts<'a>(stype: &SemanticType, int_graph: &'a IntGraph) -> Vec<STypeMount<'a>> {
    let mut mounts = Vec::new();

    for n in int_graph.graph.iter_nodes_by_label(&stype.class_uri) {
        for e in n.iter_outgoing_edges(&int_graph.graph) {
            if e.label == stype.predicate {
                mounts.push(STypeMount { edge: e });
            }
        }
    }

    return mounts
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::tests::*;
    use assembling::learning::elimination::discovery::{get_started_eliminated_2nodes, convert_to_constraint_nodes};
    use settings::conf_search::ConstraintSpace;

    #[test]
    pub fn test_find_all_mounts() {
        let int_graph = load_int_graph();
        let stype = get_semantic_type("crm:E39_Actor--karma:classLink", 1.0);
        let mounts = find_all_mounts(&stype, &int_graph);
        assert_eq!(mounts, vec![
            STypeMount { edge: int_graph.graph.get_edge_by_id(6) },
            STypeMount { edge: int_graph.graph.get_edge_by_id(68) },
        ])
    }

    #[test]
    pub fn test_get_merge_plans() {
        let (sm, int_graph, g, bijection) = get_test_data();
        let args: IntTreeSearchArgs = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);
        let sub_g = IntSubGraph::new(&g, &bijection, &args.int_graph);

        // cannot connect to man-made-object identifier
        let mount = STypeMount { edge: args.int_graph.graph.get_edge_by_id(39) };
        assert_eq!(get_merge_plan_into_tree(&sub_g, &mount, &args), vec![]);

        // cannot connect to man-made-object object uri
        let mount = STypeMount { edge: args.int_graph.graph.get_edge_by_id(22) };
        assert_eq!(get_merge_plan_into_tree(&sub_g, &mount, &args), vec![]);
        assert_eq!(get_merge_plan_into_mount(&sub_g, &mount, &args), vec![MergePlan::merge_into_mount(&mount, vec![30])]);

        // have path to production technique
        let mount = STypeMount { edge: args.int_graph.graph.get_edge_by_id(41) };
        assert_eq!(get_merge_plan_into_tree(&sub_g, &mount, &args), vec![MergePlan::merge_into_tree(&mount, vec![98])]);

        // duplication path should have been filtered out so that BeginOfExistence only go through
        // actor not production - actor
        let mount = STypeMount { edge: args.int_graph.graph.get_edge_by_id(79) };
        assert_eq!(get_merge_plan_into_tree(&sub_g, &mount, &args), vec![MergePlan::merge_into_tree(&mount, vec![87])]);
    }

    #[test]
    pub fn test_plan_merge_into_mount() {
        let (sm, int_graph, g, bijection) = get_test_data();
        let args: IntTreeSearchArgs = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);
        let sub_g = IntSubGraph::new(&g, &bijection, &args.int_graph);

        let mount = STypeMount { edge: args.int_graph.graph.get_edge_by_id(22) };
        let attr = get_attribute(0, "objectURI", &["crm:E22_Man-Made_Object--karma:classLink"], 0.0);
        let plans = MergePlan::find_merge_plans(&sub_g, &attr, &args);
        assert_eq!(plans, vec![
            MergePlan::merge_into_mount(&mount, vec![30])
        ]);

        let (new_g, new_bijection) = plans[0].proceed(&sub_g, &attr, &args);
        assert_eq!(new_g, quick_graph(&[
            "crm:E12_Production1--crm:P14_carried_out_by--crm:E39_Actor1",
            "crm:E12_Production1--karma:classLink--productionURI::d",
            "crm:E39_Actor1--karma:classLink--artistURI::d",
            "crm:E22_Man-Made_Object1--karma:classLink--objectURI::d",
            "crm:E22_Man-Made_Object1--crm:P108i_was_produced_by--crm:E12_Production1",
        ]).0);
        assert_eq!(new_bijection, Bijection::from_pairs(vec![0, 4, 27, 17, 1, 33]));
    }

    #[test]
    pub fn test_plan_merge_into_tree_and_triangle() {
        let (sm, int_graph, g, bijection) = get_test_data();

        let args: IntTreeSearchArgs = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);
        let sub_g = IntSubGraph::new(&g, &bijection, &args.int_graph);

        // test find all merged plans
        let attr = get_attribute(0, "classification", &["crm:E55_Type--rdfs:label"], 0.0);
        let plans = MergePlan::find_merge_plans(&sub_g, &attr, &args);
        let mount = STypeMount { edge: args.int_graph.graph.get_edge_by_id(41) };
        assert_eq!(plans, vec![
            MergePlan::merge_into_tree(&mount, vec![98]),
            MergePlan::merge_triangle(&mount, vec![30], vec![48, 49]),
        ]);

        let (new_g, new_bijection) = plans[0].proceed(&sub_g, &attr, &args);
        assert_eq!(new_g, quick_graph(&[
            "crm:E12_Production1--crm:P14_carried_out_by--crm:E39_Actor1",
            "crm:E12_Production1--karma:classLink--productionURI::d",
            "crm:E39_Actor1--karma:classLink--artistURI::d",
            "crm:E12_Production1--crm:P32_used_general_technique--crm:E55_Type1",
            "crm:E55_Type1--rdfs:label--classification::d"
        ]).0);
        assert_eq!(new_bijection, Bijection::from_pairs(vec![0, 4, 27, 17, 40, 50]));
        let (new_g, new_bijection) = plans[1].proceed(&sub_g, &attr, &args);
        assert_eq!(new_g, quick_graph(&[
            "crm:E12_Production1--crm:P14_carried_out_by--crm:E39_Actor1",
            "crm:E12_Production1--karma:classLink--productionURI::d",
            "crm:E39_Actor1--karma:classLink--artistURI::d",
            "crm:E22_Man-Made_Object1--crm:P108i_was_produced_by--crm:E12_Production1",
            "crm:E22_Man-Made_Object1--crm:P41i_was_classified_by--crm:E17_Type_Assignment1",
            "crm:E17_Type_Assignment1--crm:P42_assigned--crm:E55_Type1",
            "crm:E55_Type1--rdfs:label--classification::d"
        ]).0);
        assert_eq!(new_bijection, Bijection::from_pairs(vec![0, 4, 27, 17, 1, 34, 40, 50]));
    }

    #[test]
    pub fn test_plan_merge_attach() {
        let int_graph = load_int_graph();
        let sm = SemanticModel::empty("".to_owned());

        let args: IntTreeSearchArgs = IntTreeSearchArgs::default(&sm, &ConstraintSpace::default(), int_graph);
        let g = quick_graph(&[
            "crm:E39_Actor1--karma:classLink--artistURI::d",
            "crm:E39_Actor1--crm:P92i_was_brought_into_existence_by--crm:E63_Beginning_of_Existence1",
            "crm:E63_Beginning_of_Existence1--crm:P4_has_time-span--crm:E52_Time-Span1",
            "crm:E52_Time-Span1--karma:classLink--birthDate::d"
        ]).0;
        let bijection = Bijection::from_pairs(vec![4, 17, 8, 7, 24]);
        let sub_g = IntSubGraph::new(&g, &bijection, &args.int_graph);

        let attr = get_attribute(0, "birthDateEarliest", &["crm:E52_Time-Span--crm:P82a_begin_of_the_begin"], 0.0);
        let plans = MergePlan::find_merge_plans(&sub_g, &attr, &args);
        let mnt0 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(9) };
        let mnt1 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(10) };
        let mnt2 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(15) };
        let mnt3 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(66) };
        let mnt4 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(71) };
        let mnt5 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(75) };

        assert_eq!(plans, vec![
            MergePlan::merge_triangle(&mnt0, vec![29], vec![28]),
            MergePlan::merge_into_tree(&mnt1, vec![25, 27]),
            MergePlan::merge_triangle(&mnt1, vec![30, 29], vec![92, 90, 27]),
            MergePlan::attach(&mnt2),
            MergePlan::merge_triangle(&mnt5, vec![30, 29], vec![83, 82]),
            MergePlan::merge_into_tree(&mnt4, vec![87, 85]),
            MergePlan::merge_into_tree(&mnt3, vec![88]),
            MergePlan::merge_triangle(&mnt3, vec![30, 29], vec![92, 89, 88]),
        ]);
        let (new_g, new_bijection) = plans[3].proceed(&sub_g, &attr, &args);
        assert_eq!(new_g, quick_graph(&[
            "crm:E39_Actor1--karma:classLink--artistURI::d",
            "crm:E39_Actor1--crm:P92i_was_brought_into_existence_by--crm:E63_Beginning_of_Existence1",
            "crm:E63_Beginning_of_Existence1--crm:P4_has_time-span--crm:E52_Time-Span1",
            "crm:E52_Time-Span1--karma:classLink--birthDate::d",
            "crm:E52_Time-Span1--crm:P82a_begin_of_the_begin--birthDateEarliest::d"
        ]).0);

        let attr = get_attribute(0, "deathDate", &["crm:E52_Time-Span--karma:classLink"], 0.0);
        let plans = MergePlan::find_merge_plans(&sub_g, &attr, &args);
        let mnt0 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(5) };
        let mnt1 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(13) }; // cannot because of re-used old links
        let mnt2 = STypeMount { edge: args.int_graph.graph.get_edge_by_id(18) };
        assert_eq!(plans, vec![
            MergePlan::merge_triangle(&mnt0, vec![29], vec![28]),
            MergePlan::merge_into_tree(&mnt2, vec![25, 27]),
            MergePlan::merge_triangle(&mnt2, vec![30, 29], vec![92, 90, 27])
        ]);
    }

    fn get_test_data() -> (SemanticModel, IntGraph, Graph, Bijection) {
        let int_graph = load_int_graph();
        let sm = SemanticModel::empty("".to_owned());
        let g = quick_graph(&[
            "crm:E12_Production1--crm:P14_carried_out_by--crm:E39_Actor1",
            "crm:E12_Production1--karma:classLink--productionURI::d",
            "crm:E39_Actor1--karma:classLink--artistURI::d"
        ]).0;
        let bijection = Bijection::from_pairs(vec![0, 4, 27, 17]);

        (sm, int_graph, g, bijection)
    }
}