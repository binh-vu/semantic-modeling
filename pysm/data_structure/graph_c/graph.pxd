# distutils: language = c++
# distutils: sources = library.cpp

from libcpp.string cimport string
from libcpp.vector cimport vector
from sparsehash.dense_hash_map cimport dense_hash_map


cdef extern from "library.h" namespace "isi":
    enum NodeType:
        CLASS_NODE = 1
        DATA_NODE = 2

    enum LinkType:
        UNSPECIFIED = 0
        UNKNOWN = 1
        URI_PROPERTY = 2
        OBJECT_PROPERTY = 3
        DATA_PROPERTY = 4

    cdef cppclass Node:
        int id
        NodeType type
        string label
        
        vector[int] incoming_link_ids
        vector[int] outgoing_link_ids

        Node(int, NodeType, string)
        bint equal(Node&)

    cdef cppclass Link:
        int id, source_id, target_id
        LinkType type
        string label

        Link(int, LinkType, string, int, int)
        bint equal(Link&)

    cdef cppclass GraphC "isi::Graph":
        bint index_node_type, index_node_label, index_link_label
        int n_nodes, n_links

        vector[Node] nodes
        vector[Link] links
        vector[int] *class_node_index
        vector[int] *data_node_index
        dense_hash_map[string, vector[int]] *node_index_label
        dense_hash_map[string, vector[int]] *link_index_label

        GraphC(bint, bint, bint, unsigned long, unsigned long)

        Node* add_new_node(NodeType, string)
        Link* add_new_link(LinkType, string, int, int)
        bint has_node_with_id(int)
        bint has_link_with_id(int)
        Node* get_node_by_id(int)
        Link* get_link_by_id(int)


cdef class GraphNode:
    cdef readonly:
        Graph _graph
    cdef:
        Node *_node

    cpdef bint is_data_node(GraphNode)
    cpdef bint is_class_node(GraphNode)
    cpdef bint equal(GraphNode, GraphNode)
    cpdef GraphLink get_first_incoming_link(GraphNode)
    # def iter_incoming_links(GraphNode)
    # def iter_outgoing_links(GraphNode)
    cpdef str get_printed_label(GraphNode, int)
    cpdef str get_dot_format(GraphNode, int)


cdef class GraphLink:
    cdef readonly:
        Graph _graph
    cdef:
        Link *_link

    # C LEVEL FUNC
    cdef Node* get_source_node_c(GraphLink)
    cdef Node* get_target_node_c(GraphLink)
    
    # PUBLIC FUNC
    cpdef bint equal(GraphLink, GraphLink)
    cpdef GraphNode get_source_node(GraphLink)
    cpdef GraphNode get_target_node(GraphLink)
    cpdef str get_printed_label(GraphLink, int)
    cpdef str get_dot_format(GraphLink, int)


cdef class Graph:
    cdef readonly:
        string name
    cdef:
        list gnodes
        list glinks
        GraphC *_graph_c

    # PRIVATE FUNC
    cdef Node* add_new_node_c(Graph, NodeType, string) except +
    cdef Link* add_new_link_c(Graph, LinkType, string, int, int) except +

    # C LEVEL FUNC
    cdef Node* get_node_by_id_c(Graph, int)
    cdef Link* get_link_by_id_c(Graph, int)
    cdef int get_n_nodes_c(Graph)
    cdef int get_n_links_c(Graph)

    # PUBLIC FUNC
    cpdef void set_name(Graph, string)
    cpdef int get_n_nodes(Graph)
    cpdef int get_n_links(Graph)
    cpdef Graph clone(Graph, int estimated_n_nodes=*, int estimated_n_links=*)

    # multi weird API to add node/link to support inheritance
    # real_add_new_node mean this is an actual function that execute the logic in graph (but it calling add_new_node_c)
    cpdef GraphNode real_add_new_node(Graph, GraphNode, NodeType, string)
    cpdef GraphLink real_add_new_link(Graph, GraphLink, LinkType, string, int, int)
    cpdef GraphNode add_new_node(Graph, NodeType, string)
    cpdef GraphLink add_new_link(Graph, LinkType, string, int, int)

    cpdef bint has_node_with_id(Graph, int)
    cpdef bint has_link_with_id(Graph, int)
    cpdef GraphNode get_node_by_id(Graph, int)
    cpdef GraphLink get_link_by_id(Graph, int)

    # Commented codes are valid python functions, since cython doesn't allow def func in .pxd file
    cpdef list iter_nodes(Graph)
    # def iter_class_nodes(Graph)
    # def iter_data_nodes(Graph)
    # def iter_nodes_by_label(Graph, string)
    cpdef list iter_links(Graph)
    # def iter_links_by_label(Graph, string)
    cpdef bint equal(self, Graph)
    cpdef dict to_dict(Graph)
    # cpdef Graph from_dict(dict)
    # def render(Graph, int, int)
    # def render2img(Graph, str, int)
    # def render2pdf(Graph, str, int)
