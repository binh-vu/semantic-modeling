# distutils: language = c++

from data_structure.graph_c.graph cimport Graph, GraphNode, GraphLink, Node, NodeType, Link, LinkType

cdef class BoundedGraphNode(GraphNode):
    cdef:
        int node_offset
        int link_offset


cdef class BoundedGraphLink(GraphLink):
    cdef:
        int node_offset
        int link_offset


cdef class CrossBorderGraphLink(GraphLink):
    cdef readonly:
        int id
        int source_id       # id in part_a or b (is already offseted)
        int target_id       # id in part a or b (is already offseted)
    cdef:
        int source_id_prime # id in part_merge (is already offseted)
        int target_id_prime # id in part_merge (is already offseted)


cdef class CrossBorderGraphNode(GraphNode):
    """Represent a node in the border of the graph, and the link connect between 2 graph is incoming/outgoing link of this node"""
    cdef readonly:
        int id
    cdef:
        int link_offset
        bint is_incoming_link
        CrossBorderGraphLink cross_border_link


cdef class IntegrationPoint:
    cdef readonly:
        bint is_incoming_link
        int link_id
        int source_id
        int target_id


cdef class MergeGraph(Graph):
    cdef:
        int part_b_n_offset
        int part_merge_n_offset
        int part_b_e_offset
        int part_merge_e_offset
        int n_nodes
        int n_links

        # offset for part_a = 0
        Graph g_part_a
        Graph g_part_b
        Graph g_part_merge

        CrossBorderGraphNode node_a  # node between part a & part merge
        CrossBorderGraphNode node_b  # node between part b & part merge
        CrossBorderGraphLink link_a  # link between part a & part merge
        CrossBorderGraphLink link_b  # link between part b & part merge

    cpdef Graph proceed_merging(MergeGraph)
