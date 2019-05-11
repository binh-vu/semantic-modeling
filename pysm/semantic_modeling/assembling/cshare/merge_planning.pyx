# distutils: language = c++

from libcpp.string cimport string
from cython.operator cimport dereference as deref, postincrement as inc
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from data_structure.graph_c.graph cimport Graph, GraphNode, Node, Link, GraphC, GraphLink
from semantic_modeling.assembling.cshare.unordered_set cimport unordered_set

from semantic_modeling.assembling.cshare.graph_explorer cimport GraphNodeHop
from semantic_modeling.assembling.cshare.graph_explorer cimport GraphExplorer
from semantic_modeling.assembling.cshare.merge_graph cimport IntegrationPoint, MergeGraph
from sparsehash.dense_hash_map cimport dense_hash_map


cdef class MergePlan:
    cdef readonly:
        Graph int_tree  # a tree/path connect 2 graphs A, B together
        IntegrationPoint int_a  #
        IntegrationPoint int_b

    def __init__(self, Graph int_tree, IntegrationPoint int_a, IntegrationPoint int_b):
        self.int_tree = int_tree
        self.int_a = int_a
        self.int_b = int_b


cdef pair[int, int] add_merge_path(Graph g_merge, GraphExplorer explorer, int x_n_id) except *:
    """A merge path is like this: A -- X_1 -- X_2 -- ... -- X_N -- B where N can vary from 0 to inf.
    Input:
        * a graph where the merge path will be added to
        * a graph explorer which contains the merge_path
        * id of X_N in a graph explorer, and add all path from X_N downback to X_1
    Output: 
        if N > 0: id of link A -- X_1 and id of X_1 in `g_merge` so that they can complete the merge path by adding A -- X_1
        if N == 0: (-1, -1)
    """
    cdef:
        GraphC* explorer_c = explorer._graph_c
        GraphNodeHop x_i = explorer.get_node_by_id(x_n_id)
        Node *x_i_c = x_i._node
        Node *x_i_1_c = NULL
        int y_i_c_id, y_i_1_c_id
        Link *link = NULL
        pair[int, int] result

    if x_i.n_hop == 0:
        result.first = -1
        result.second = -1
        return result

    y_i_c_id = g_merge.add_new_node(x_i_c.type, x_i_c.label).id
    if x_i.n_hop > 0:
        # -1 so that we only go to X_1 not A
        for i in range(x_i.n_hop - 1):
            # must have incoming_link_ids
            link_id = x_i_c.incoming_link_ids.at(0)
            link = &explorer_c.links[link_id]
            x_i_1_c = &explorer_c.nodes[link.source_id]

            y_i_1_c_id = g_merge.add_new_node(x_i_1_c.type, x_i_1_c.label).id
            g_merge.add_new_link(link.type, link.label, y_i_1_c_id, y_i_c_id)
            x_i_c = x_i_1_c
            y_i_c_id = y_i_1_c_id

        result.first = x_i_c.incoming_link_ids.at(0)
        result.second = y_i_c_id
    else:
        for i in range(- x_i.n_hop - 1):
            link_id = x_i_c.outgoing_link_ids.at(0)
            link = &explorer_c.links[link_id]
            x_i_1_c = &explorer_c.nodes[link.target_id]

            y_i_1_c_id = g_merge.add_new_node(x_i_1_c.type, x_i_1_c.label).id
            g_merge.add_new_link(link.type, link.label, y_i_c_id, y_i_1_c_id)
            x_i_c = x_i_1_c
            y_i_c_id = y_i_1_c_id

        result.first = x_i_c.outgoing_link_ids.at(0)
        result.second = y_i_c_id
    return result


def py_add_merge_path(Graph g_merge, GraphExplorer explorer, int x_n_id):
    return add_merge_path(g_merge, explorer, x_n_id)


cdef Graph combine_merge_path(Graph g_merge_a, Graph g_merge_b, int x_n_id, int y_n_id):
    """Combine 2 merge paths together, each of them epresented by a graph of linear chain X_1 -- X_2 -- ... -- X_N and Y_1 -- Y_2 -- ... -- Y_N.
    The points to connect 2 chains are X_N and Y_N, output: X_1 -- X_2 -- ... X_N -- Y_N_1 -- ... Y_1  
    """
    cdef:
        Graph g_merge = g_merge_a.clone()
        GraphC *g_merge_b_c = g_merge_b._graph_c
        int i
        int *id_map = <int *> PyMem_Malloc(g_merge_b_c.n_nodes * sizeof(int))
        Node *node
        Link *link

    if not id_map:
        raise MemoryError()

    for i in range(g_merge_b_c.n_nodes):
        node = &g_merge_b_c.nodes[i]
        if node.id != y_n_id:
            id_map[i] = g_merge.real_add_new_node(GraphNode(), node.type, node.label).id
        else:
            id_map[i] = x_n_id

    for i in range(g_merge_b_c.n_links):
        link = &g_merge_b_c.links[i]
        g_merge.real_add_new_link(GraphLink(), link.type, link.label, id_map[link.source_id], id_map[link.target_id])

    # remember to free id_map
    PyMem_Free(id_map)

    return g_merge


def py_combine_merge_path(Graph g_merge_a, Graph g_merge_b, int x_n_id, int y_n_id):
    return combine_merge_path(g_merge_a, g_merge_b, x_n_id, y_n_id)


cdef GraphNode get_root_node(Graph tree):
    cdef int i

    for i in range(tree.get_n_nodes_c()):
        if tree._graph_c.nodes[i].incoming_link_ids.size() == 0:
            return tree.gnodes[i]

    return None


cdef list make_plan4case23_subfunc(Graph treeA, GraphExplorer treeAsearch, GraphNode rootB):
    """Generate plan that merge tree B into treeA (A -- B)"""
    cdef:
        GraphNodeHop x_n
        GraphNodeHop b
        GraphLink link
        list plans = []

    for b in treeAsearch.iter_gt0hop_nodes_by_label(rootB.label):
        link = b.get_first_incoming_link()
        x_n = link.get_source_node()
        if x_n.n_hop == 0:
            g_merge = Graph()
            g_merge.add_new_node(x_n.type, x_n.label)  # A' = 0
            g_merge.add_new_node(b.type, b.label)   # B' = 1
            g_merge.add_new_link(link.type, link.label, 0, 1)  # (A' -> B')

            int_a = IntegrationPoint(False, x_n.id, 0, 1)  # source=A, target=B'
            int_b = IntegrationPoint(True, 0, 0, rootB.id)  # source=A', target=rootB
            plans.append(MergePlan(g_merge, int_a, int_b))
        else:
            g_merge = Graph()
            link_a_id, x_1_id_prime = add_merge_path(g_merge, treeAsearch, x_n.id)
            link_a = treeAsearch.get_link_by_id(link_a_id)
            node_a = link_a.get_source_node()
            b_id_prime = g_merge.add_new_node(b.type, b.label).id
            a_id_prime = g_merge.add_new_node(node_a.type, node_a.label).id

            link_a_id_prime = g_merge.add_new_link(link_a.type, link_a.label, a_id_prime, x_1_id_prime).id
            link_b_id_prime = g_merge.add_new_link(link.type, link.label, 0, b_id_prime).id  # X'_n = 0
            int_a = IntegrationPoint(False, node_a.id, link_a_id_prime, x_1_id_prime)
            int_b = IntegrationPoint(True, 0, link_b_id_prime, rootB.id)
            plans.append(MergePlan(g_merge, int_a, int_b))

    return plans


cdef list make_plan4case23_specialcase(Graph treeA, GraphExplorer treeAsearch, GraphNode rootB, GraphExplorer treeBsearch):
    cdef:
        int x_1_id_prime, link_a_id, link_a_id_prime

        list plans = []
    # note that node in Graph share same id like node in GraphExplorer
    for link in treeBsearch.get_node_by_id(rootB.id).iter_incoming_links():
        pB = link.get_source_node()
        for x_n in treeAsearch.iter_gt0hop_nodes_by_label(pB.label):
            g_merge = Graph()
            link_a_id, x_1_id_prime = add_merge_path(g_merge, treeAsearch, x_n.id)
            link_a = treeAsearch.get_link_by_id(link_a_id)
            node_a = link_a.get_source_node()
            a_id_prime = g_merge.add_new_node(node_a.type, node_a.label).id
            b_id_prime = g_merge.add_new_node(rootB.type, rootB.label).id
            link_a_id_prime = g_merge.add_new_link(link_a.type, link_a.label, a_id_prime, x_1_id_prime).id
            link_b_id_prime = g_merge.add_new_link(link.type, link.label, 0, b_id_prime).id
            int_a = IntegrationPoint(False, node_a.id, link_a_id_prime, x_1_id_prime)
            int_b = IntegrationPoint(True, 0, link_b_id_prime, rootB.id)

            plans.append(MergePlan(g_merge, int_a, int_b))

        for x_n in treeA.iter_nodes_by_label(pB.label):
            g_merge = Graph()
            g_merge.add_new_node(pB.type, pB.label)
            g_merge.add_new_node(rootB.type, rootB.label)
            g_merge.add_new_link(link.type, link.label, 0, 1)
            int_a = IntegrationPoint(False, x_n.id, 0, 1)
            int_b = IntegrationPoint(True, 0, 0, rootB.id)

            plans.append(MergePlan(g_merge, int_a, int_b))
    return plans


cdef list make_plan4case23(Graph treeA, Graph treeB, GraphExplorer treeAsearch, GraphExplorer treeBsearch):
    cdef:
        GraphNode rootA = get_root_node(treeA)
        GraphNode rootB = get_root_node(treeB)

    if rootB.is_data_node() and rootA.is_class_node():
        return make_plan4case23_specialcase(treeA, treeAsearch, rootB, treeBsearch)
    elif rootA.is_data_node() and rootB.is_class_node():
        return make_plan4case23_specialcase(treeB, treeBsearch, rootA, treeAsearch)

    return make_plan4case23_subfunc(treeA, treeAsearch, rootB) \
        + make_plan4case23_subfunc(treeB, treeBsearch, rootA)

cdef list make_plan4case1(Graph treeA, Graph treeB, GraphExplorer treeAsearch, GraphExplorer treeBsearch):
    cdef:
        GraphNode rootA, rootB, node
        unsigned int i, id, x_1_id_prime, y_1_id_prime, link_a_id, link_b_id, a_id_prime, b_id_prime, link_a_id_prime, link_b_id_prime
        Node *node_b, *node_a
        Link *link_a, *link_b
        unordered_set[string] a_ancestors
        unordered_set[string] common_ancestors
        dense_hash_map[string, vector[int]] a_plan_inputs
        dense_hash_map[string, vector[int]] b_plan_inputs
        Graph g_merge
        list merge_plan_a = []
        list merge_plan_b = []
        list plans = []

    rootA = get_root_node(treeA)
    rootB = get_root_node(treeB)

    # get common ancestors
    for i in range(treeAsearch.lt0hop_nodes.size()):
        id = treeAsearch.lt0hop_nodes[i]
        a_ancestors.insert(treeAsearch._graph_c.nodes[id].label)

    for i in range(treeBsearch.lt0hop_nodes.size()):
        id = treeBsearch.lt0hop_nodes[i]
        node_b = &treeBsearch._graph_c.nodes[id]
        if a_ancestors.find(node_b.label) != a_ancestors.end():
            # this is an common ancestor
            common_ancestors.insert(node_b.label)

    # make plans from those ancestors
    for common_ancestor in common_ancestors:
        # make plan from a => common ancestor
        # TODO: should directly call C/C++ api
        merge_plan_a = []
        merge_plan_b = []
        for node in treeAsearch.iter_lt0hop_nodes_by_label(common_ancestor):
            g_merge = Graph()
            link_a_id, x_1_id_prime = add_merge_path(g_merge, treeAsearch, node.id)
            merge_plan_a.append((g_merge, link_a_id, x_1_id_prime))

        for node in treeBsearch.iter_lt0hop_nodes_by_label(common_ancestor):
            g_merge = Graph()
            link_b_id, y_1_id_prime = add_merge_path(g_merge, treeBsearch, node.id)
            merge_plan_b.append((g_merge, link_b_id, y_1_id_prime))

        for g_merge_a, link_a_id, x_1_id_prime in merge_plan_a:
            link_a = &treeAsearch._graph_c.links[link_a_id]
            node_a = &treeAsearch._graph_c.nodes[link_a.target_id]
            for g_merge_b, link_b_id, y_1_id_prime in merge_plan_b:
                link_b = &treeBsearch._graph_c.links[link_b_id]
                node_b = &treeBsearch._graph_c.nodes[link_b.target_id]

                # node that X_n_prime & Y_n_prime always 0 due to the fact that id is assigned continuously
                if y_1_id_prime != 0:
                    # NOTE: we add link B - Y1' to g_merge, but because when we add Y1 to g_merge, it has different id
                    # the new id is the old id shifted by |g_merge_a.n_nodes| - 1
                    # there is a case where g_merge_b have only one node, then y_1_id_prime is actually node 0, which is
                    # removed by merging into X_N in g_merge_a
                    y_1_id_prime = y_1_id_prime + g_merge_a.get_n_nodes() - 1

                g_merge = combine_merge_path(g_merge_a, g_merge_b, 0, 0)
                a_id_prime = g_merge.add_new_node(node_a.type, node_a.label).id
                b_id_prime = g_merge.add_new_node(node_b.type, node_b.label).id
                link_a_id_prime = g_merge.add_new_link(link_a.type, link_a.label, x_1_id_prime, a_id_prime).id
                link_b_id_prime = g_merge.add_new_link(link_b.type, link_b.label, y_1_id_prime, b_id_prime).id

                plans.append(MergePlan(
                    g_merge,
                    IntegrationPoint(True, x_1_id_prime, link_a_id_prime, node_a.id),
                    IntegrationPoint(True, y_1_id_prime, link_b_id_prime, node_b.id)))

    return plans

def py_make_plan4case23(Graph treeA, Graph treeB, GraphExplorer treeAsearch, GraphExplorer treeBsearch):
    return make_plan4case23(treeA, treeB, treeAsearch, treeBsearch)

def py_make_plan4case1(Graph treeA, Graph treeB, GraphExplorer treeAsearch, GraphExplorer treeBsearch):
    return make_plan4case1(treeA, treeB, treeAsearch, treeBsearch)

def make_merge_plans(Graph treeA, Graph treeB, GraphExplorer treeAsearch, GraphExplorer treeBsearch):
    return make_plan4case1(treeA, treeB, treeAsearch, treeBsearch) + make_plan4case23(treeA, treeB, treeAsearch, treeBsearch)




