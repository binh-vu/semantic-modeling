#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, List

from data_structure import GraphNodeType, Graph, GraphLinkType
from semantic_modeling.assembling.triple_adviser import TripleAdviser
from semantic_modeling.assembling.cshare.graph_explorer import GraphNodeHop, GraphExplorer


class GraphExplorerBuilder(object):
    def __init__(self, triple_adviser: TripleAdviser, max_data_node_hop: int, max_class_node_hop: int) -> None:
        super().__init__()
        self.max_data_node_hop: int = max_data_node_hop
        self.max_class_node_hop: int = max_class_node_hop
        self.triple_adviser: TripleAdviser = triple_adviser

        self.explored_data_node: Dict[bytes, GraphExplorer] = {}
        self.explored_class_node: Dict[bytes, GraphExplorer] = {}

    def build(self, g: Graph) -> GraphExplorer:
        # TODO: can make it more efficient by giving estimation to graph explorer
        g_explorer = GraphExplorer()
        for node in g.iter_nodes():
            g_explorer.real_add_new_node(GraphNodeHop(0), node.type, node.label)
        for link in g.iter_links():
            g_explorer.add_new_link(link.type, link.label, link.source_id, link.target_id)

        self.explore(g_explorer)
        return g_explorer

    def explore(self, g: GraphExplorer) -> None:
        for i in range(g.get_n_nodes()):
            curr_node = g.get_node_by_id(i)
            # if current node label is not cached, then init and cache it
            if curr_node.is_class_node():
                if curr_node.label not in self.explored_class_node:
                    self.explored_class_node[curr_node.label] = self._explore_node(curr_node, self.max_class_node_hop)

                explored_graph = self.explored_class_node[curr_node.label]
            else:
                if curr_node.label not in self.explored_data_node:
                    self.explored_data_node[curr_node.label] = self._explore_node(curr_node, self.max_data_node_hop)

                explored_graph = self.explored_data_node[curr_node.label]

            if curr_node.n_incoming_links == 0:
                # to maintain tree structure, it can only expand if it doesn't have any parent
                self._explore_cached_incoming_node(g, curr_node, explored_graph)
            self._explore_cached_outgoing_node(g, curr_node, explored_graph)

    def _explore_cached_incoming_node(self,
                                      updating_graph: GraphExplorer,
                                      node: GraphNodeHop,
                                      explored_graph: GraphExplorer) -> None:
        """Add all possible incoming node from `node` to `updating_graph`"""
        current_hop = [explored_graph.get_node_by_id(0)]  # node 0 always in central hop
        id_map: Dict[int, int] = {0: node.id}
        next_hop = []

        while len(current_hop) > 0:
            for node in current_hop:
                for incoming_link in node.iter_incoming_links():
                    source_node: GraphNodeHop = incoming_link.get_source_node()
                    new_node = updating_graph.real_add_new_node(
                        GraphNodeHop(source_node.n_hop), source_node.type, source_node.label)
                    id_map[source_node.id] = new_node.id
                    updating_graph.add_new_link(incoming_link.type, incoming_link.label,
                                                id_map[incoming_link.source_id], id_map[incoming_link.target_id])
                    next_hop.append(source_node)

            current_hop, next_hop = next_hop, []

    def _explore_cached_outgoing_node(self,
                                      updating_graph: GraphExplorer,
                                      node: GraphNodeHop,
                                      explored_graph: GraphExplorer) -> None:
        """Add all possible outgoing node from `node` to `updating_graph`"""
        current_hop = [explored_graph.get_node_by_id(0)]
        id_map: Dict[int, int] = {0: node.id}
        next_hop = []

        while len(current_hop) > 0:
            for current_node in current_hop:
                for outgoing_link in current_node.iter_outgoing_links():
                    target_node: GraphNodeHop = outgoing_link.get_target_node()
                    new_node = updating_graph.real_add_new_node(
                        GraphNodeHop(target_node.n_hop), target_node.type, target_node.label)
                    id_map[target_node.id] = new_node.id

                    updating_graph.add_new_link(outgoing_link.type, outgoing_link.label,
                                                id_map[outgoing_link.source_id], id_map[outgoing_link.target_id])

                    next_hop.append(target_node)
            current_hop, next_hop = next_hop, []

    def _explore_node(self, node: GraphNodeHop, max_hop: int) -> GraphExplorer:
        g = GraphExplorer()
        current_hop: List[GraphNodeHop] = [g.real_add_new_node(GraphNodeHop(0), node.type, node.label)]
        next_hop = []

        # there are many possible links between two nodes, we want only one link between 2 nodes, so
        # we are going to have so many same labeled nodes in explore graph
        # add outgoing nodes, positive hop
        for hop in range(max_hop):
            for curr_node in current_hop:
                for link_label, target_label in self.triple_adviser.get_pred_objs(curr_node.label):
                    next_hop_node: GraphNodeHop = g.real_add_new_node(
                        GraphNodeHop(hop + 1), GraphNodeType.CLASS_NODE, target_label)
                    g.add_new_link(GraphLinkType.UNSPECIFIED, link_label, curr_node.id, next_hop_node.id)
                    next_hop.append(next_hop_node)

            current_hop, next_hop = next_hop, []

        current_hop: List[GraphNodeHop] = [g.get_node_by_id(0)]
        next_hop = []

        # add incoming nodes, negative hop
        for hop in range(max_hop):
            for curr_node in current_hop:
                for source_label, link_label in self.triple_adviser.get_subj_preds(curr_node.label):
                    next_hop_node: GraphNodeHop = g.real_add_new_node(
                        GraphNodeHop(-hop - 1), GraphNodeType.CLASS_NODE, source_label)
                    g.add_new_link(GraphLinkType.UNSPECIFIED, link_label, next_hop_node.id, curr_node.id)
                    next_hop.append(next_hop_node)
            current_hop, next_hop = next_hop, []

        return g

