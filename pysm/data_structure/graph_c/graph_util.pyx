# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector
from data_structure.graph_c.graph cimport Graph, GraphNode, GraphLink, Node, Link, GraphC


cdef string tree_hashing(GraphC *g, Node *node, int incoming_link_id, vector[int] &visited_index) except +:
    cdef:
        list children_texts = []
        int link_id
        string result

    if visited_index[node.id] == incoming_link_id:
        # re-visit via same link, cycle detection
        return ""

    visited_index[node.id] = incoming_link_id
    if node.outgoing_link_ids.size() == 0:
        return node.label

    for link_id in node.outgoing_link_ids:
        result = tree_hashing(g, &g.nodes[g.links[link_id].target_id], link_id, visited_index)
        children_texts.append(b"%s-%s" % (g.links[link_id].label, result))

    return b"(%s:%s)" % (node.label, b",".join(sorted(children_texts)))


cpdef string to_hashable_string(Graph g):
    # we don't guarantee that it will return consistent result if graph has cycle
    cdef:
        list roots = []
        Node *node
        GraphC *gc = g._graph_c
        int i
        vector[int] visited_index = vector[int](g.get_n_nodes_c(), -2)

    for i in range(g.get_n_nodes_c()):
        node = &gc.nodes[i]
        if node.incoming_link_ids.size() == 0:
            roots.append(tree_hashing(gc, node, -1, visited_index))

    return b",".join(sorted(roots))