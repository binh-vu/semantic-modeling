# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from data_structure.graph_c.graph cimport Graph, GraphNode, GraphLink, Node, NodeType, Link, LinkType



cdef class BoundedGraphNode(GraphNode):
    def __init__(self, int node_offset, int link_offset) -> None:
        self.node_offset = node_offset
        self.link_offset = link_offset

    @property
    def id(self): return self._node.id + self.node_offset

    def equal(self, *args, **kwargs) -> bool:
        raise AttributeError("Not support `equal` func")

    cpdef GraphLink get_first_incoming_link(self):
        if self._node.incoming_link_ids.size() > 0:
            return self._graph.get_link_by_id(self._node.incoming_link_ids[0] + self.link_offset)
        else:
            return None

    def iter_incoming_links(self):
        cdef int i
        for i in range(self._node.incoming_link_ids.size()):
            yield self._graph.get_link_by_id(self._node.incoming_link_ids[i] + self.link_offset)

    def iter_outgoing_links(self):
        cdef int i
        for i in range(self._node.outgoing_link_ids.size()):
            yield self._graph.get_link_by_id(self._node.outgoing_link_ids[i] + self.link_offset)


cdef class BoundedGraphLink(GraphLink):
    def __init__(self, int node_offset, int link_offset) -> None:
        self.node_offset = node_offset
        self.link_offset = link_offset

    @property
    def id(self): return self._link.id + self.link_offset
    @property
    def source_id(self): return self._link.source_id + self.node_offset
    @property
    def target_id(self): return self._link.target_id + self.node_offset

    def equal(self, *args, **kwargs):
        raise AttributeError("Not support `equal` func")

    cpdef GraphNode get_source_node(self):
        return self._graph.get_node_by_id(self._link.source_id + self.node_offset)

    cpdef GraphNode get_target_node(self):
        return self._graph.get_node_by_id(self._link.target_id + self.node_offset)


cdef class CrossBorderGraphLink(GraphLink):
    def __init__(self, int source_id, int source_id_prime, int target_id, int target_id_prime, int link_id):
        self.id = link_id
        self.source_id = source_id  # id in part_a or b (is already offset)
        self.target_id = target_id  # id in part a or b (is already offset)
        self.source_id_prime = source_id_prime  # id in part_merge (is already offset)
        self.target_id_prime = target_id_prime  # id in part_merge (is already offset)

    def equal(self, *args, **kwargs):
        raise AttributeError("Not support `equal` func")

    cpdef GraphNode get_source_node(self):
        return self._graph.get_node_by_id(self.source_id)

    cpdef GraphNode get_target_node(self):
        return self._graph.get_node_by_id(self.target_id)


cdef class CrossBorderGraphNode(GraphNode):
    """Represent a node in the border of the graph, and the link connect between 2 graph is incoming/outgoing link of this node"""
    def __init__(self, int link_offset, bint is_incoming_link, CrossBorderGraphLink cross_border_link):
        self.link_offset = link_offset
        self.is_incoming_link = is_incoming_link
        self.cross_border_link = cross_border_link
        if is_incoming_link:
            self.id = cross_border_link.target_id
        else:
            self.id = cross_border_link.source_id

    @property
    def n_incoming_links(self):
        if self.is_incoming_link:
            return self._node.incoming_link_ids.size() + 1
        return self._node.incoming_link_ids.size()

    @property
    def n_outgoing_links(self):
        if self.is_incoming_link:
            return self._node.outgoing_link_ids.size()
        return self._node.outgoing_link_ids.size() + 1

    def equal(self, *args, **kwargs):
        raise AttributeError("Not support `equal` func")

    cpdef GraphLink get_first_incoming_link(self):
        if self.is_incoming_link:
            return self.cross_border_link
        elif self._node.incoming_link_ids.size() > 0:
            return self._graph.get_link_by_id(self._node.incoming_link_ids[0] + self.link_offset)
        else:
            return None

    def iter_incoming_links(self):
        cdef int i
        for i in range(self._node.incoming_link_ids.size()):
            yield self._graph.get_link_by_id(self._node.incoming_link_ids[i] + self.link_offset)

        if self.is_incoming_link:
            yield self.cross_border_link

    def iter_outgoing_links(self):
        cdef int i
        for i in range(self._node.outgoing_link_ids.size()):
            yield self._graph.get_link_by_id(self._node.outgoing_link_ids[i] + self.link_offset)

        if not self.is_incoming_link:
            yield self.cross_border_link


cdef class IntegrationPoint:
    def __init__(self, bint is_incoming_link, int source_id, int link_id, int target_id):
        # all the id is local (not the final id we get after merge 2 graph together)
        # either source_id or target_id is in part_a/part_b, the remain id is in part_merge
        self.is_incoming_link = is_incoming_link
        self.source_id = source_id
        self.link_id = link_id  # link_id always id in part_merge
        self.target_id = target_id

    def to_dict(self):
        return {"is_incoming_link": self.is_incoming_link, "source_id": self.source_id, "link_id": self.link_id, "target_id": self.target_id}

cdef class MergeGraph(Graph):
    def __init__(self, bint index_node_type=False, bint index_node_label=False, bint index_link_label=False, int estimated_n_nodes=0, int estimated_n_links=0, string name=b"graph", Graph g_part_a=None, Graph g_part_b=None, Graph g_part_merge=None, IntegrationPoint point_a=None, IntegrationPoint point_b=None):
        """When we construct this graph, notice that there are two nodes in part_merge got removed,
        a_prime and b_prime. Those two nodes are integration node, merged to a, and b in part_a and part_b, respectively.

        Because of the removal, node id won't be in a continuously range. it would be easier if we assume those two nodes
        are stored at the end of part_merge node's array
        """
        self.part_b_n_offset = g_part_a.get_n_nodes_c()
        self.part_merge_n_offset = self.part_b_n_offset + g_part_b.get_n_nodes_c()
        self.part_b_e_offset = g_part_a.get_n_links_c()
        self.part_merge_e_offset = self.part_b_e_offset + g_part_b.get_n_links_c()

        self.g_part_a = g_part_a
        self.g_part_b = g_part_b
        self.g_part_merge = g_part_merge

        cdef:
            int n_links_of_part_merge = g_part_merge._graph_c.n_links
            int n_nodes_of_part_merge = g_part_merge.get_n_nodes_c() - 2

        self.n_nodes = self.part_merge_n_offset + n_nodes_of_part_merge
        self.n_links = self.part_merge_e_offset + n_links_of_part_merge

        # setup cross border node
        if point_a.is_incoming_link:
            # compute source_id because part_a doesn't have source node
            if n_links_of_part_merge > 1:
                # because part_merge is actually a linear-chain, so when number of links > 1, point_a.source
                # is not an integration point
                source_id = point_a.source_id + self.part_merge_n_offset
            else:
                # point_a.source is an integration point, so source_id is point_b.source_id moved by some units
                source_id = point_b.source_id + self.part_b_n_offset

            self.link_a = CrossBorderGraphLink(
                source_id,
                point_a.source_id + self.part_merge_n_offset,
                point_a.target_id,
                g_part_merge.get_link_by_id_c(point_a.link_id).target_id + self.part_merge_n_offset,
                point_a.link_id + self.part_merge_e_offset
            )
            self.link_a._graph = self
            if self.link_a.target_id_prime < self.n_nodes:
                assert False, "Merge point in part_merge should be stored at the end: (%s >= %s)" % (self.link_a.target_id_prime, self.n_nodes)
        else:
            # compute target_id because part_a doesn't have target node
            if n_links_of_part_merge > 1:
                # because part_merge is actually a linear-chain, so when number of links > 1, point_a.target
                # is not an integration point
                target_id = point_a.target_id + self.part_merge_n_offset
            else:
                # point_a.target is an integration point, so target_id is point_b.target_id moved by some units
                target_id = point_b.target_id + self.part_b_n_offset

            self.link_a = CrossBorderGraphLink(
                point_a.source_id,
                g_part_merge.get_link_by_id_c(point_a.link_id).source_id + self.part_merge_n_offset,
                target_id,
                point_a.target_id + self.part_merge_n_offset,
                point_a.link_id + self.part_merge_e_offset
            )
            self.link_a._graph = self
            if self.link_a.source_id_prime < self.n_nodes:
                assert False, "Merge point in part_merge should be stored at the end: (%s >= %s)" % (self.link_a.source_id_prime, self.n_nodes)
        self.link_a._link = g_part_merge.get_link_by_id_c(point_a.link_id)

        if point_b.is_incoming_link:
            # compute source_id because part_b doesn't have source node, same like point_a
            if n_links_of_part_merge > 1:
                source_id = point_b.source_id + self.part_merge_n_offset
            else:
                source_id = self.link_a.source_id

            self.link_b = CrossBorderGraphLink(
                source_id,
                point_b.source_id + self.part_merge_n_offset,
                point_b.target_id + self.part_b_n_offset,
                g_part_merge.get_link_by_id_c(point_b.link_id).target_id + self.part_merge_n_offset,
                point_b.link_id + self.part_merge_e_offset
            )
            self.link_b._graph = self
            if self.link_b.target_id_prime < self.n_nodes:
                assert False, "Merge point in part_merge should be stored at the end: (%s >= %s)" % (self.link_b.target_id_prime, self.n_nodes)
        else:
            # compute target_id because part_b doesn't have target node, same like point_a
            if n_links_of_part_merge > 1:
                target_id = point_b.target_id + self.part_merge_n_offset
            else:
                target_id = self.link_a.target_id

            self.link_b = CrossBorderGraphLink(
                point_b.source_id + self.part_b_n_offset,
                g_part_merge.get_link_by_id_c(point_b.link_id).source_id + self.part_merge_n_offset,
                target_id,
                point_b.target_id + self.part_merge_n_offset,
                point_b.link_id + self.part_merge_e_offset
            )
            self.link_b._graph = self
            if self.link_b.source_id_prime < self.n_nodes:
                assert False, "Merge point in part_merge should be stored at the end: (%s >= %s)" % (self.link_b.source_id_prime, self.n_nodes)
        self.link_b._link = g_part_merge.get_link_by_id_c(point_b.link_id)

        self.node_a = CrossBorderGraphNode(0, point_a.is_incoming_link, self.link_a)
        self.node_a._graph = self
        if point_a.is_incoming_link:
            self.node_a._node = &g_part_a._graph_c.nodes[point_a.target_id]
        else:
            self.node_a._node = &g_part_a._graph_c.nodes[point_a.source_id]

        self.node_b = CrossBorderGraphNode(self.part_b_e_offset, point_b.is_incoming_link, self.link_b)
        self.node_b._graph = self
        if point_b.is_incoming_link:
            self.node_b._node = &g_part_b._graph_c.nodes[point_b.target_id]
        else:
            self.node_b._node = &g_part_b._graph_c.nodes[point_b.source_id]

    @staticmethod
    def create(Graph g_part_a, Graph g_part_b, Graph g_part_merge, IntegrationPoint point_a, IntegrationPoint point_b):
        # part merge usually doesn't have many nodes, and loop through the nodes usually yield better performance
        # so we let user control via part_merge index variable (if enable, mean we will use index)
        cdef:
            bint index_node_type = g_part_a._graph_c.index_node_type and g_part_b._graph_c.index_node_type # and g_part_merge._graph_c.index_node_type
            bint index_node_label = g_part_a._graph_c.index_node_label and g_part_b._graph_c.index_node_label # and g_part_merge._graph_c.index_node_label
            bint index_link_label = g_part_a._graph_c.index_link_label and g_part_b._graph_c.index_link_label # and g_part_merge._graph_c.index_link_label

        return MergeGraph(index_node_type=index_node_type, index_node_label=index_node_label,
            index_link_label=index_link_label, g_part_a=g_part_a, g_part_b=g_part_b,
            g_part_merge=g_part_merge, point_a=point_a, point_b=point_b)

    # ##########################################################################
    # C LEVEL FUNC

    # ##########################################################################
    # PUBLIC FUNC
    @property
    def index_node_type(self): return self._graph_c.index_node_type
    @property
    def index_node_label(self): return self._graph_c.index_node_label
    @property
    def index_link_label(self): return self._graph_c.index_link_label

    def set_name(self, name: bytes):
        raise AttributeError("MergeGraph doesn't support set_name")

    cpdef int get_n_nodes(self):
        return self.n_nodes

    cpdef int get_n_links(self):
        return self.n_links

    def real_add_new_node(self, node: GraphNode, type: NodeType, label: bytes) -> GraphNode:
        raise AttributeError("MergeGraph doesn't support real_add_new_node")

    def real_add_new_link(self, link: GraphLink, type: LinkType, label: bytes, source_id: int, target_id: int) -> GraphLink:
        raise AttributeError("MergeGraph doesn't support real_add_new_link")

    def add_new_node(self, type: NodeType, label: bytes) -> GraphNode:
        raise AttributeError("MergeGraph doesn't support add_new_node")

    def add_new_link(self, type: LinkType, label: bytes, source_id: int, target_id: int) -> GraphLink:
        raise AttributeError("MergeGraph doesn't support add_new_link")

    cpdef bint has_node_with_id(self, int id):
        return 0 <= id < self.n_nodes

    cpdef bint has_link_with_id(self, int id):
        return 0 <= id < self.n_links

    cpdef GraphNode get_node_by_id(self, int id):
        cdef:
            int node_offset = 0, link_offset = 0
            Node* _ptr
            BoundedGraphNode node

        if id < self.part_b_n_offset:
            if id == self.node_a.id:
                # is cross_border node
                return self.node_a
            # is bounded node
            _ptr = &self.g_part_a._graph_c.nodes[id]
        elif id < self.part_merge_n_offset:
            if id == self.node_b.id:
                # is cross_border node
                return self.node_b
            node_offset = self.part_b_n_offset
            link_offset = self.part_b_e_offset
            _ptr = &self.g_part_b._graph_c.nodes[id - node_offset]
        else:
            # doesn't handle a case when they access id outside of n_nodes (including the two deleted nodes)
            node_offset = self.part_merge_n_offset
            link_offset = self.part_merge_e_offset
            _ptr = &self.g_part_merge._graph_c.nodes[id - node_offset]

        node = BoundedGraphNode(node_offset, link_offset)
        node._node = _ptr
        node._graph = self
        return node

    cpdef GraphLink get_link_by_id(self, int id):
        cdef:
            int node_offset = 0, link_offset = 0
            Link* _ptr
            BoundedGraphLink link

        if id < self.part_b_e_offset:
            _ptr = &self.g_part_a._graph_c.links[id]
        elif id < self.part_merge_e_offset:
            node_offset = self.part_b_n_offset
            link_offset = self.part_b_e_offset
            _ptr = &self.g_part_b._graph_c.links[id - link_offset]
        else:
            if id == self.link_a.id:
                return self.link_a
            if id == self.link_b.id:
                return self.link_b
            node_offset = self.part_merge_n_offset
            link_offset = self.part_merge_e_offset
            _ptr = &self.g_part_merge._graph_c.links[id - link_offset]

        link = BoundedGraphLink(node_offset, link_offset)
        link._link = _ptr
        link._graph = self
        return link

    cpdef Graph proceed_merging(self):
        cdef:
            Graph g = Graph(self.index_node_type, self.index_node_label, self.index_link_label, self.n_nodes, self.n_links)
            GraphNode node
            GraphLink link
            int i

        for i in range(self.n_nodes):
            node = self.get_node_by_id(i)
            g.add_new_node(node.type, node.label)

        for i in range(self.n_links):
            link = self.get_link_by_id(i)
            g.add_new_link(link.type, link.label, link.source_id, link.target_id)

        return g


    def iter_nodes(self):
        cdef int i = 0
        for i in range(self.n_nodes):
            yield self.get_node_by_id(i)

    def iter_links(self):
        cdef int i = 0
        for i in range(self.n_links):
            yield self.get_link_by_id(i)

    def iter_class_nodes(self):
        cdef:
            unsigned int i
            int id
            vector[int] *id_array
            vector[Node] *nodes

        if not self._graph_c.index_node_type:
            # to avoiding calling assert everytime
            assert self._graph_c.index_node_type

        id_array = self.g_part_a._graph_c.class_node_index
        for i in range(id_array.size()):
            id = deref(id_array)[i]
            yield self.get_node_by_id(id)

        id_array = self.g_part_b._graph_c.class_node_index
        for i in range(id_array.size()):
            id = deref(id_array)[i] + self.part_b_n_offset
            yield self.get_node_by_id(id)

        # if part_merge enable index, then we use it, if not use for-loop (user tell us it loop will be faster)
        if self.g_part_merge._graph_c.index_node_type:
            id_array = self.g_part_merge._graph_c.class_node_index
            for i in range(id_array.size()):
                id = deref(id_array)[i] + self.part_merge_n_offset
                if id < self.n_nodes:
                    yield self.get_node_by_id(id)
        else:
            nodes = &self.g_part_merge._graph_c.nodes
            for i in range(nodes.size() - 2):
                if deref(nodes)[i].type == NodeType.CLASS_NODE:
                    yield self.get_node_by_id(deref(nodes)[i].id + self.part_merge_n_offset)

    def iter_data_nodes(self):
        cdef:
            unsigned int i
            int id
            vector[int] *id_array

        if not self._graph_c.index_node_type:
            # to avoiding calling assert everytime
            assert self._graph_c.index_node_type

        id_array = self.g_part_a._graph_c.data_node_index
        for i in range(id_array.size()):
            id = deref(id_array)[i]
            yield self.get_node_by_id(id)

        id_array = self.g_part_b._graph_c.data_node_index
        for i in range(id_array.size()):
            id = deref(id_array)[i] + self.part_b_n_offset
            yield self.get_node_by_id(id)

    def iter_nodes_by_label(self, label: bytes):
        cdef:
            unsigned int i
            int id
            string lbl
            vector[int] *id_array
            vector[Node] *nodes

        if not self._graph_c.index_node_label:
            # to avoiding calling assert everytime
            assert self._graph_c.index_node_label

        lbl = label
        id_array = &deref(self.g_part_a._graph_c.node_index_label)[lbl]
        for i in range(id_array.size()):
            id = deref(id_array)[i]
            yield self.get_node_by_id(id)

        id_array = &deref(self.g_part_b._graph_c.node_index_label)[lbl]
        for i in range(id_array.size()):
            id = deref(id_array)[i] + self.part_b_n_offset
            yield self.get_node_by_id(id)

        # if part_merge enable index, then we use it, if not use for-loop (user tell us it loop will be faster)
        if self.g_part_merge._graph_c.index_node_label:
            id_array = &deref(self.g_part_merge._graph_c.node_index_label)[lbl]
            for i in range(id_array.size()):
                id = deref(id_array)[i] + self.part_merge_n_offset
                if id < self.n_nodes:
                    yield self.get_node_by_id(id)
        else:
            nodes = &self.g_part_merge._graph_c.nodes
            for i in range(nodes.size() - 2):
                if deref(nodes)[i].label.compare(lbl) == 0:
                    yield self.get_node_by_id(deref(nodes)[i].id + self.part_merge_n_offset)

    def iter_links_by_label(self, label: bytes):
        cdef:
            unsigned int i
            int id
            string lbl
            vector[int] *id_array
            vector[Node] *nodes

        if not self._graph_c.index_link_label:
            # to avoiding calling assert everytime
            assert self._graph_c.index_link_label

        lbl = label

        id_array = &deref(self.g_part_a._graph_c.link_index_label)[lbl]
        for i in range(id_array.size()):
            id = deref(id_array)[i]
            yield self.get_link_by_id(id)

        id_array = &deref(self.g_part_b._graph_c.link_index_label)[lbl]
        for i in range(id_array.size()):
            id = deref(id_array)[i] + self.part_b_e_offset
            yield self.get_link_by_id(id)

        # if part_merge enable index, then we use it, if not use for-loop (user tell us it loop will be faster)
        if self.g_part_merge._graph_c.index_link_label:
            id_array = &deref(self.g_part_merge._graph_c.link_index_label)[lbl]
            for i in range(id_array.size()):
                id = deref(id_array)[i] + self.part_merge_e_offset
                yield self.get_link_by_id(id)
        else:
            links = &self.g_part_merge._graph_c.links
            for i in range(links.size()):
                if deref(links)[i].label.compare(lbl) == 0:
                    yield self.get_link_by_id(deref(links)[i].id + self.part_merge_e_offset)
