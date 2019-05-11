#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, TYPE_CHECKING

from data_structure import Graph, GraphNode, GraphLink, graph2dict, dict2graph
from semantic_modeling.karma.karma_link import _dict_camel_to_snake, KarmaGraphLink
from semantic_modeling.karma.karma_node import KarmaGraphNode
from semantic_modeling.utilities.ontology import Ontology

if TYPE_CHECKING:
    from semantic_modeling.karma.karma import KarmaSourceColumn


class KarmaGraph(Graph):
    def __init__(self,
                 index_node_type=False,
                 index_node_label=False,
                 index_link_label=False,
                 estimated_n_nodes=24,
                 estimated_n_links=23,
                 name=b"graph") -> None:
        super().__init__(index_node_type, index_node_label, index_link_label, estimated_n_nodes, estimated_n_links,
                         name)
        self.model: 'KarmaModel' = None

    def to_dict(self):
        return graph2dict(self, None, KarmaGraphNode.node2meta, KarmaGraphLink.link2meta)

    @staticmethod
    def from_dict(obj: dict) -> 'KarmaGraph':
        return dict2graph(obj, KarmaGraph, KarmaGraphNode, KarmaGraphLink, lambda g: {}, KarmaGraphNode.meta2args,
                          KarmaGraphLink.meta2args)

    def set_model(self, model: 'KarmaModel') -> 'KarmaGraph':
        self.model = model
        self.set_name(model.id.encode('utf-8'))
        return self

    def to_graph(self) -> Graph:
        g = Graph(
            index_node_type=True,
            index_node_label=True,
            index_link_label=True,
            estimated_n_nodes=self.get_n_nodes(),
            estimated_n_links=self.get_n_links(),
            name=self.name)
        for n in self.iter_nodes():
            g.add_new_node(n.type, n.label)
        for e in self.iter_links():
            g.add_new_link(e.type, e.label, e.source_id, e.target_id)
        return g

    @staticmethod
    def from_karma_model(name: bytes, graph: dict, ontology: Ontology,
                         id2columns: Dict[str, 'KarmaSourceColumn']) -> Tuple['KarmaGraph', dict, dict]:
        karma_graph = KarmaGraph(True, True, True, len(graph['nodes']), len(graph['links']), name)
        node_idmap, link_idmap = {}, {}

        for node in graph['nodes']:
            node = _dict_camel_to_snake(node)
            node_idmap[node['id']] = len(node_idmap)
            assert karma_graph.real_add_new_node(*KarmaGraphNode.from_karma_model(
                node, ontology, id2columns)).id == node_idmap[node['id']]

            node['id'] = node_idmap[node['id']]

        for link in graph['links']:
            link = _dict_camel_to_snake(link)
            link_idmap[link['id']] = len(link_idmap)

            source_id, link_label, target_id = link['id'].split('---')
            link['id'] = link_idmap[link['id']]
            link['source_id'] = node_idmap[source_id]
            link['target_id'] = node_idmap[target_id]

            assert karma_graph.real_add_new_link(*KarmaGraphLink.from_karma_model(link, ontology)).id == link['id']

        return karma_graph, node_idmap, link_idmap
