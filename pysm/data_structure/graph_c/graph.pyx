# distutils: language = c++

from cython.operator cimport dereference as deref
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from semantic_modeling.algorithm.string import auto_wrap
from data_structure.utilities import render_graph, graph2img, graph2pdf, graph2dict, dict2graph
from sparsehash.dense_hash_map cimport dense_hash_map


cdef class GraphNode:

    @property
    def id(self): return self._node.id
    @property
    def type(self): return self._node.type
    @property
    def label(self): return self._node.label
    @property
    def n_incoming_links(self): return self._node.incoming_link_ids.size()
    @property
    def n_outgoing_links(self): return self._node.outgoing_link_ids.size()

    cpdef bint is_data_node(self):
        return self._node.type == NodeType.DATA_NODE

    cpdef bint is_class_node(self):
        return self._node.type == NodeType.CLASS_NODE

    cpdef bint equal(self, GraphNode another):
        return self._node.equal(deref(another._node))

    cpdef GraphLink get_first_incoming_link(self):
        if self._node.incoming_link_ids.size() > 0:
            return self._graph.get_link_by_id(self._node.incoming_link_ids[0])
        else:
            return None

    def iter_incoming_links(self):
        cdef int i
        for i in range(self._node.incoming_link_ids.size()):
            yield self._graph.get_link_by_id(self._node.incoming_link_ids[i])

    def iter_outgoing_links(self):
        cdef int i
        for i in range(self._node.outgoing_link_ids.size()):
            yield self._graph.get_link_by_id(self._node.outgoing_link_ids[i])

    cpdef str get_printed_label(self, int max_text_width):
        return auto_wrap("N%03d: %s" % (self._node.id, self._node.label.decode('utf-8')), max_text_width)

    cpdef str get_dot_format(self, int max_text_width):
        label = self.get_printed_label(max_text_width).encode('unicode_escape').decode('utf-8')
        if self._node.type == NodeType.CLASS_NODE:
            return '"%s"[style="filled",color="white",fillcolor="lightgray",label="%s"];' % (self._node.id, label)

        return '"%s"[shape="plaintext",style="filled",fillcolor="gold",label="%s"];' % (self._node.id, label)
    

cdef class GraphLink:
    
    # C LEVEL FUNC
    cdef Node* get_source_node_c(self):
        return &self._graph._graph_c.nodes[self._link.source_id]

    cdef Node* get_target_node_c(self):
        return &self._graph._graph_c.nodes[self._link.target_id]
    
    # PUBLIC FUNC
    @property
    def id(self): return self._link.id
    @property
    def type(self): return self._link.type
    @property
    def label(self): return self._link.label
    @property
    def source_id(self): return self._link.source_id
    @property
    def target_id(self): return self._link.target_id

    cpdef bint equal(self, GraphLink another):
        return another is not None and \
            self._link.id == another._link.id and \
            self._link.type == another._link.type and \
            self._link.label == another._link.label and \
            self._link.source_id == another._link.source_id and \
            self._link.target_id == another._link.target_id

    cpdef GraphNode get_source_node(self):
        return self._graph.gnodes[self._link.source_id]

    cpdef GraphNode get_target_node(self):
        return self._graph.gnodes[self._link.target_id]

    cpdef str get_printed_label(self, int max_text_width):
        return auto_wrap("L%03d: %s" % (self._link.id, self._link.label.decode('utf-8')), max_text_width)

    cpdef str get_dot_format(self, int max_text_width):
        label = self.get_printed_label(max_text_width).encode('unicode_escape').decode('utf-8')
        return '"%s" -> "%s"[color="brown",fontcolor="black",label="%s"];' % (self._link.source_id, self._link.target_id, label)


cdef class Graph:

    def __cinit__(self, bint index_node_type=False, bint index_node_label=False, bint index_link_label=False, int estimated_n_nodes=24, int estimated_n_links=23, string name=b"graph", *args, **kwargs):
        self._graph_c = new GraphC(index_node_type, index_node_label, index_link_label, estimated_n_nodes, estimated_n_links)

    def __init__(self, bint index_node_type=False, bint index_node_label=False, bint index_link_label=False, int estimated_n_nodes=24, int estimated_n_links=23, string name=b"graph"):
        self.name = name
        self.gnodes = []
        self.glinks = []

    def __dealloc__(self):
        del self._graph_c

    # ##########################################################################
    # PRIVATE FUNC
    cdef Node* add_new_node_c(self, NodeType type, string label) except +:
        cdef:
            GraphNode gnode
            Node *node
            int i
            bint is_resize

        is_resize = self._graph_c.nodes.capacity() == self._graph_c.n_nodes
        node = self._graph_c.add_new_node(type, label)
        if is_resize:
            # hit the capacity, vector will be resize, we need to let every nodes aware of it
            i = 0
            for gnode in self.gnodes:
                gnode._node = &self._graph_c.nodes[i]
                i += 1

        return node

    cdef Link* add_new_link_c(self, LinkType type, string label, int source_id, int target_id) except +:
        cdef:
            GraphLink glink
            Link *link
            int i
            bint is_resize
            
        is_resize = self._graph_c.links.capacity() == self._graph_c.n_links
        link = self._graph_c.add_new_link(type, label, source_id, target_id)
        if is_resize:
            # hit the capacity, vector will be resize, we need to let every links aware of it
            i = 0
            for glink in self.glinks:
                glink._link = &self._graph_c.links[i]
                i += 1
        
        return link

    # ##########################################################################
    # C LEVEL FUNC
    cdef Node* get_node_by_id_c(self, int id):
        return &self._graph_c.nodes[id]

    cdef Link* get_link_by_id_c(self, int id):
        return &self._graph_c.links[id]

    cdef int get_n_nodes_c(self):
        return self._graph_c.n_nodes

    cdef int get_n_links_c(self):
        return self._graph_c.n_links

    # ##########################################################################
    # PUBLIC FUNC

    @property
    def index_node_type(self): return self._graph_c.index_node_type
    @property
    def index_node_label(self): return self._graph_c.index_node_label
    @property
    def index_link_label(self): return self._graph_c.index_link_label

    cpdef void set_name(self, string name):
        self.name = name

    cpdef int get_n_nodes(self):
        return self._graph_c.n_nodes

    cpdef int get_n_links(self):
        return self._graph_c.n_links

    cpdef Graph clone(self, int estimated_n_nodes=-1, int estimated_n_links=-1):
        if estimated_n_nodes == -1 or estimated_n_links == -1:
            estimated_n_nodes = self._graph_c.n_nodes
            estimated_n_links = self._graph_c.n_links

        cdef:
            Graph g = Graph(self.index_node_type, self.index_node_label, self.index_link_label, estimated_n_nodes, estimated_n_links, self.name)
            int i
            Node *node
            Link *link

        for i in range(self._graph_c.n_nodes):
            node = &self._graph_c.nodes[i]
            g.real_add_new_node(GraphNode(), node.type, node.label)

        for i in range(self._graph_c.n_links):
            link = &self._graph_c.links[i]
            g.real_add_new_link(GraphLink(), link.type, link.label, link.source_id, link.target_id)
        return g

    cpdef GraphNode real_add_new_node(self, GraphNode gnode, NodeType type, string label):
        gnode._node = self.add_new_node_c(type, label)
        gnode._graph = self
        self.gnodes.append(gnode)
        return gnode

    cpdef GraphLink real_add_new_link(self, GraphLink glink, LinkType type, string label, int source_id, int target_id):
        glink._link = self.add_new_link_c(type, label, source_id, target_id)
        glink._graph = self
        self.glinks.append(glink)
        return glink

    cpdef GraphNode add_new_node(self, NodeType type, string label):
        return self.real_add_new_node(GraphNode(), type, label)

    cpdef GraphLink add_new_link(self, LinkType type, string label, int source_id, int target_id):
        return self.real_add_new_link(GraphLink(), type, label, source_id, target_id)

    cpdef bint has_node_with_id(self, int id):
        return self._graph_c.has_node_with_id(id)

    cpdef bint has_link_with_id(self, int id):
        return self._graph_c.has_link_with_id(id)

    cpdef GraphNode get_node_by_id(self, int id):
        return self.gnodes[id]

    cpdef GraphLink get_link_by_id(self, int id):
        return self.glinks[id]

    cpdef list iter_nodes(self):
        return self.gnodes

    cpdef list iter_links(self):
        return self.glinks

    def iter_class_nodes(self):
        cdef:
            unsigned int i
            vector[int] *id_array 

        if not self._graph_c.index_node_type:
            # to avoiding calling assert everytime
            assert self._graph_c.index_node_type, "Must be indexed before invoking"

        id_array = self._graph_c.class_node_index
        for i in range(id_array.size()):
            yield self.gnodes[<int> deref(id_array)[i]]

    def iter_data_nodes(self):
        cdef:
            unsigned int i
            vector[int] *id_array             

        if not self._graph_c.index_node_type:
            # to avoiding calling assert everytime
            assert self._graph_c.index_node_type, "Must be indexed before invoking"

        id_array = self._graph_c.data_node_index
        for i in range(id_array.size()):
            yield self.gnodes[<int> deref(id_array)[i]]

    def iter_nodes_by_label(self, string label):
        cdef:
            unsigned int i
            vector[int] *id_array

        if not self._graph_c.index_node_label:
            # to avoiding calling assert everytime
            assert self._graph_c.index_node_label, "Must be indexed before invoking"

        id_array = &deref(self._graph_c.node_index_label)[label]
        for i in range(id_array.size()):
            yield self.gnodes[<int> deref(id_array)[i]]

    def iter_links_by_label(self, string label):
        cdef:
            unsigned int i
            vector[int] *id_array

        if not self._graph_c.index_link_label:
            # to avoiding calling assert everytime
            assert self._graph_c.index_link_label, "Must be indexed before invoking"

        id_array = &deref(self._graph_c.link_index_label)[label]
        for i in range(id_array.size()):
            yield self.glinks[<int> deref(id_array)[i]]

    cpdef bint equal(self, Graph another):
        if self._graph_c.n_nodes != another._graph_c.n_nodes or self._graph_c.n_links != another._graph_c.n_links:
            return False

        for node in self.gnodes:
            if not node.equal(another.gnodes[node.id]):
                return False

        for link in self.glinks:
            if not link.equal(another.glinks[link.id]):
                return False
        return True

    cpdef dict to_dict(self):
        return graph2dict(self)

    @staticmethod
    def from_dict(dict obj) -> Graph:
        return dict2graph(obj, Graph, GraphNode, GraphLink)

    def render(self, int dpi=50, int max_text_width=15) -> None:
        render_graph(self, dpi, max_text_width)

    def render2img(self, f_output, int max_text_width=15) -> None:
        graph2img(self, f_output, max_text_width)

    def render2pdf(self, f_output, int max_text_width=15) -> None:
        graph2pdf(self, f_output, max_text_width)

    # implement pickling, notice that it doesn't work with sub-classes
    def __getnewargs__(self):
        return self.index_node_type, self.index_node_label, self.index_link_label, self.get_n_nodes_c(), self.get_n_links_c()

    def __getstate__(self):
        nodes = [
            (node.id, node.type, node.label) for node in self.iter_nodes()
        ]
        links = [
            (link.id, link.type, link.label, link.source_id, link.target_id) for link in self.iter_links()
        ]
        return nodes, links, self.name

    def __setstate__(self, state):
        nodes, links, name = state
        assert self.get_n_nodes_c() == 0

        # it doesn't invoke __init__, so we have to do it here
        self.name = name
        self.gnodes = []
        self.glinks = []

        for node in nodes:
            self.add_new_node(node[1], node[2])
        for link in links:
            self.add_new_link(link[1], link[2], link[3], link[4])