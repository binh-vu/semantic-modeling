#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
from collections import Counter
from typing import List, Dict

from data_structure import Graph, GraphNode, GraphLink
from semantic_modeling.karma.karma_graph import KarmaGraph
from semantic_modeling.karma.karma_link import KarmaGraphLink
from semantic_modeling.karma.karma_node import KarmaSemanticType, KarmaGraphNode
from semantic_modeling.utilities.ontology import Ontology


class SemanticType(object):

    def __init__(self, domain: str, type: str, confidence_score: float) -> None:
        self.domain = domain
        self.type = type
        self.confidence_score = confidence_score

    def to_dict(self):
        return {
            "domain": self.domain,
            "type": self.type,
            "confidence_score": self.confidence_score
        }

    @staticmethod
    def from_dict(obj: dict):
        return SemanticType(obj['domain'], obj['type'], obj['confidence_score'])


class Attribute(object):

    def __init__(self, id: int, label: str, semantic_types: List[SemanticType]):
        self.id = id  # id of a column in semantic model, may have a corresponding node (same id) in the graph
        self.label = label  # label must be unique
        self.semantic_types: List[SemanticType] = semantic_types

    def to_dict(self):
        return {"id": self.id, "label": self.label, "semantic_types": [st.to_dict() for st in self.semantic_types]}

    @staticmethod
    def from_dict(obj: dict) -> 'Attribute':
        return Attribute(obj['id'], obj['label'], [SemanticType.from_dict(st) for st in obj['semantic_types']])


class SemanticModel(object):

    def __init__(self, id: str, attrs: List[Attribute], graph: Graph, disable_check: bool=False) -> None:
        """
        :param id: id of the semantic model
        :param graph: its graph representation
        """
        self.id = id
        self.attrs: List[Attribute] = attrs
        self.label2attrs: Dict[str, Attribute] = {attr.label: attr for attr in attrs}
        self.graph = graph

        assert disable_check or len(self.label2attrs) == len(self.attrs), "We shouldn't have two same-label attributes"

    def get_attr_by_label(self, name: str) -> Attribute:
        return self.label2attrs[name]

    def has_attr(self, name: str) -> bool:
        return name in self.label2attrs

    def to_dict(self) -> dict:
        return {"name": self.id, "attrs": [a.to_dict() for a in self.attrs], "graph": self.graph.to_dict()}

    @staticmethod
    def from_dict(obj: dict) -> 'SemanticModel':
        return SemanticModel(obj['name'], [Attribute.from_dict(a) for a in obj['attrs']], Graph.from_dict(obj['graph']))

    def to_karma_json_model(self, ont: Ontology) -> dict:
        from semantic_modeling.karma.karma import KarmaSourceColumn, KarmaMappingToSourceColumn, KarmaModel

        id2attrs = {a.id: a for a in self.attrs}
        source_columns = [KarmaSourceColumn(attr.id, attr.id, attr.label) for attr in self.attrs]
        mapping_to_source_columns = [KarmaMappingToSourceColumn(attr.id, attr.id) for attr in self.attrs]

        karma_graph = KarmaGraph(True, True, True, self.graph.get_n_nodes(), self.graph.get_n_links())

        for node in self.graph.iter_nodes():
            if node.is_data_node():
                if node.id not in id2attrs:  # literal type
                    user_semantic_types = []
                else:
                    user_semantic_types = [
                        KarmaSemanticType(str(node.id), stype.domain, stype.type, "UNKNOWN", stype.confidence_score)
                        for stype in id2attrs[node.id].semantic_types
                    ]
                karma_node = KarmaGraphNode(user_semantic_types, [], None, node.id not in id2attrs)
            else:
                karma_node = KarmaGraphNode([], [], None, False)
            karma_graph.real_add_new_node(karma_node, node.type, node.label)

        for link in self.graph.iter_links():
            karma_link = KarmaGraphLink(0)
            karma_graph.real_add_new_link(karma_link, link.type, link.label, link.source_id, link.target_id)

        return KarmaModel(self.id, None, source_columns, mapping_to_source_columns,
                          karma_graph).to_normalized_json_model(ont)
