# distutils: language = c++

from data_structure.graph_c.graph cimport GraphNode, Graph
from libcpp.vector cimport vector
from libcpp.string cimport string
from sparsehash.dense_hash_map cimport dense_hash_map

cdef class GraphNodeHop(GraphNode):
    cdef readonly:
        int n_hop


cdef class GraphExplorer(Graph):
    cdef:
        vector[int] eq0hop_nodes
        vector[int] gt0hop_nodes
        vector[int] lt0hop_nodes
        dense_hash_map[string, vector[int]] lt0hop_index_node_label
        dense_hash_map[string, vector[int]] gt0hop_index_node_label
