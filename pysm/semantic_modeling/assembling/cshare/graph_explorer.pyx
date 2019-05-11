# distutils: language = c++
from data_structure import GraphLink, dict2graph, graph2dict
from data_structure.graph_c.graph cimport GraphNode, NodeType, Graph
from libcpp.string cimport string
from cython.operator cimport dereference as deref


cdef class GraphNodeHop(GraphNode):
    def __init__(self, int n_hop):
        self.n_hop = n_hop

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return {"n_hop": obj["n_hop"]}

    @staticmethod
    def node2meta(self: GraphNodeHop) -> dict:
        return {"n_hop": self.n_hop}

    def get_dot_format(self, max_text_width: int):
        label = self.get_printed_label(max_text_width).encode('unicode_escape').decode('utf-8') + " >> N_HOP: %d" % (self.n_hop,)
        if self.is_class_node():
            if self.n_hop == 0:
                return '"%s"[style="filled",color="white",fillcolor="gold",label="%s"];' % (self.id, label)
            return '"%s"[style="filled",color="white",fillcolor="lightgray",label="%s"];' % (self.id, label)
        else:
            if self.n_hop == 0:
                return '"%s"[style="filled",shape="plaintext",color="white",fillcolor="gold",label="%s"];' % (self.id,
                                                                                                              label)
            return '"%s"[style="filled",shape="plaintext",color="white",fillcolor="lightgray",label="%s"];' % (self.id,
                                                                                                               label)

cdef class GraphExplorer(Graph):

    def __cinit__(self,
                  bint index_node_type=False, bint index_node_label=False, bint index_link_label=False,
                  int estimated_n_nodes=24, int estimated_n_links=23, string name=b"graph",
                  int estimated_n_0hop=4, int estimated_n_gt0hop=8, int estimated_n_lt0hop=12, *args, **kwargs):
        self.eq0hop_nodes.reserve(estimated_n_0hop)
        self.gt0hop_nodes.reserve(estimated_n_gt0hop)
        self.lt0hop_nodes.reserve(estimated_n_lt0hop)
        self.lt0hop_index_node_label.set_empty_key("empty-key-afedc06d-ab5f-45a2-aa11-073229c26f25")
        self.gt0hop_index_node_label.set_empty_key("empty-key-afedc06d-ab5f-45a2-aa11-073229c26f25")

    def __init__(self, bint index_node_type=False, bint index_node_label=False, bint index_link_label=False,
                  int estimated_n_nodes=24, int estimated_n_links=23, string name=b"graph",
                  int estimated_n_0hop=4, int estimated_n_gt0hop=8, int estimated_n_lt0hop=12, *args, **kwargs):
        super(GraphExplorer, self).__init__(index_node_type, index_node_label, index_link_label, estimated_n_nodes, estimated_n_links, name)

    cpdef GraphNode real_add_new_node(self, GraphNode gnode, NodeType type, string label):
        """gnode is actually GraphNodeHop, cython don't let us override that"""
        super(GraphExplorer, self).real_add_new_node(gnode, type, label)
        if gnode.n_hop == 0:
            self.eq0hop_nodes.push_back(gnode._node.id)
        elif gnode.n_hop > 0:
            self.gt0hop_nodes.push_back(gnode._node.id)
            if self.gt0hop_index_node_label.find(gnode._node.label) == self.gt0hop_index_node_label.end():
                self.gt0hop_index_node_label[gnode._node.label]
            self.gt0hop_index_node_label[gnode._node.label].push_back(gnode._node.id)
        else:
            self.lt0hop_nodes.push_back(gnode._node.id)
            if self.lt0hop_index_node_label.find(gnode._node.label) == self.lt0hop_index_node_label.end():
                self.lt0hop_index_node_label[gnode._node.label]
            self.lt0hop_index_node_label[gnode._node.label].push_back(gnode._node.id)

        return gnode

    @staticmethod
    def graph2meta(self: GraphExplorer) -> dict:
        return {
            "estimated_n_0hop": self.eq0hop_nodes.size(),
            "estimated_n_gt0hop": self.gt0hop_nodes.size(),
            "estimated_n_lt0hop": self.lt0hop_nodes.size()
        }

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return {name: obj[name] for name in ["estimated_n_0hop", "estimated_n_gt0hop", "estimated_n_lt0hop"]}

    def to_dict(self) -> dict:
        return graph2dict(self, GraphExplorer.graph2meta, GraphNodeHop.node2meta)

    @staticmethod
    def from_dict(obj: dict) -> GraphExplorer:
        return dict2graph(obj, GraphExplorer, GraphNodeHop, GraphLink, GraphExplorer.meta2args, GraphNodeHop.meta2args)

    def iter_eq0hop_nodes(self):
        cdef:
            unsigned int i

        for i in range(self.eq0hop_nodes.size()):
            yield self.gnodes[<int> self.eq0hop_nodes[i]]

    def iter_gt0hop_nodes(self):
        cdef:
            unsigned int i

        for i in range(self.gt0hop_nodes.size()):
            yield self.gnodes[<int> self.gt0hop_nodes[i]]

    def iter_lt0hop_nodes(self):
        cdef:
            unsigned int i

        for i in range(self.lt0hop_nodes.size()):
            yield self.gnodes[<int> self.lt0hop_nodes[i]]

    def iter_lt0hop_nodes_by_label(self, label: string):
        cdef:
            unsigned int i
            vector[int] *id_array
        id_array = &self.lt0hop_index_node_label[label]
        for i in range(id_array.size()):
            yield self.gnodes[<int> deref(id_array)[i]]

    def iter_gt0hop_nodes_by_label(self, label: string):
        cdef:
            unsigned int i
            vector[int] *id_array
        id_array = &self.gt0hop_index_node_label[label]
        for i in range(id_array.size()):
            yield self.gnodes[<int> deref(id_array)[i]]

    # implement pickling, notice that it doesn't work with sub-classes
    def __getnewargs__(self):
        return self.index_node_type, self.index_node_label, self.index_link_label, self.get_n_nodes_c(), self.get_n_links_c(), self.name, self.eq0hop_nodes.size(), self.gt0hop_nodes.size(), self.lt0hop_nodes.size()

    def __getstate__(self):
        nodes = [
            (node.id, node.type, node.label, node.n_hop) for node in self.iter_nodes()
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
            self.real_add_new_node(GraphNodeHop(node[3]), node[1], node[2])
        for link in links:
            self.add_new_link(link[1], link[2], link[3], link[4])