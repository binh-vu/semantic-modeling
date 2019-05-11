This is an algorithm to discover next states based on the integration graph (space of all possible semantic models) in MohsenJWS 2015.

The idea is graphs may have multiple mapping into integration graph. And we retain this mapping information, so which part of
integration graph it belong to.

# DataStructures

IntTreeSearchNode:
    graph: Graph,
    bijections: Mapping from graph node to int graph

IntTreeSearchArgs:
    n_empty_hop: usize,
    int_graph: IntGraph
    filter: Func<MergePlan> -> bool

# Discover algorithm:

Input:
    search_node: IntTreeSearchNode,
    search_args: IntTreeSearchArgs

Output:
    Vec<IntTreeSearchNode>


next_nodes = []

for bijection in search_node.bijections:
    for attr in search_node.remained_attrs.iter():
        possible_mount = find_possible_mount(search_node, attr)
        if possible_mount is not None:
            plan = get_plan_connect_two_tree(search_node, possible_mount, attr)
            if args.filter(plan):
                next_nodes.add(plan.proceed())


return next_nodes

# FIND_MERGE_PLANs

Input:
    graph: Graph,
    bijection: Bijection
    attribute: Attribute
    int_graph: IntGraph

Output:
    A list of plans to connect graph and the attributes

// loop through each semantic types, and find a mount in IntTreeGraph
// then travel upward to find a path to match this
let tree_a = SubIntGraph(graph, bijection)
let plans = []

for stype in attributes.semantic_types:
    for mount in FIND_ALL_MOUNTs(stype, int_graph):
        if HAS_PATH_BETWEEN_TWO_NODES(mount.class_node, tree_a.root):
            // tree_a is subtree of tree_b
            path = GET_MOUNT_PATH_FROM_TREE(tree_a, mount)
            plans.add(path)

        if HAS_PATH_BETWEEN_TWO_NODES(tree_a.root, mount.class_node):
            // tree_b is subtree of tree_a
            path = GET_MOUNT_PATH_FROM_TREE(tree_a, tree_b)
            plans.add(path)

        if NO_PATH_BETWEEN_TWO_NODES(tree_a.root, mount.class_node):
            // find all path that connect tree_a.root and mount.class_node
            merge_plan that connect two root nodes
            plans.add(path)

return plans

# HAS_PATH_BETWEEN_TWO_NODES(source, target, int_graph) -> bool

# FIND_ALL_MOUNTs(stype: SemanticType, int_graph: IntGraph):

Output:
    List of Mount

mounts = []

for node in int_graph.iter_nodes_by_label(stype.class_uri):
    for e in node.iter_outgoing_edges():
        if e == stype.predicate:
            mounts add Mount { class_node: node.id, predicate: e.id, data_node: e.target_id }

return mounts

# GET_MOUNT_PATH_FROM_TREE(tree_a: SubIntGraph, mount: Mount):

A mount path is a list of edges in IntGraph from root of tree_a to class_node of mount

Output:
    Vec<IntEdge>

current_node = mount.class_node
result = []

loop:
    if current_node has no parent:
        panic!(Invalid mount because there is no path tree to mount)

    edge = current_node.parent_edge
    parent = current_node.parent

    result.add(edge)

    if parent in tree_a:
        // hit node in tree_a, we done
        break

return result

# GET_MOUNT_PATH_TO_TREE(tree: SubIntGraph, mount: Mount):

A mount path is a list of edges in IntGraph from root of tree_a to class_node of mount

Output:
    Vec<IntEdge>

current_node = root of tree
result = []

loop:
    if current_node has no parent:
        panic!(Invalid mount because there is no path from mount to tree a)

    edge = current_node.edge
    parent = current_node.parent

    result.add(edge)

    if parent == mount.class_node:
        // hit node in mount, we done
        break

return result
