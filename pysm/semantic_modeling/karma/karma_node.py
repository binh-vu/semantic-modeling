#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, TYPE_CHECKING

from pyutils.list_utils import unique_values

from data_structure import GraphNode, GraphNodeType
from semantic_modeling.karma.karma_link import _dict_camel_to_snake
from semantic_modeling.utilities.ontology import Ontology

if TYPE_CHECKING:
    from semantic_modeling.karma.karma import KarmaSourceColumn


class KarmaSemanticType(object):

    def __init__(self, h_node_id: Union[str, int], domain: str, type: str, origin: str, confidence_score: float) -> None:
        self.h_node_id = h_node_id
        self.domain: str = domain
        self.type: str = type
        self.origin: str = origin  # who set this
        self.confidence_score: float = confidence_score

    def is_same_type(self, other: 'KarmaSemanticType') -> bool:
        return self.domain == other.domain and self.type == other.type

    def to_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def from_dict(obj: dict) -> 'KarmaSemanticType':
        return KarmaSemanticType(**obj)

    def __repr__(self):
        return 'SemanticType(h_node_id=%s, origin=%s, confident_score=%.4f, domain=%s, type=%s)' % (
            self.h_node_id, self.origin, self.confidence_score, self.domain, self.type)

    def get_hashing_id(self) -> str:
        """Return a unique string represent this object"""
        return str(self)


class KarmaGraphNode(GraphNode):

    def __init__(self,
                 user_semantic_types: List[KarmaSemanticType],
                 learned_semantic_types: List[KarmaSemanticType],
                 literal_type: Optional[str],
                 is_literal_node: bool) -> None:
        """ the semantic type labeled by the user. contains domain, type of the data node """
        self.user_semantic_types: List[KarmaSemanticType] = user_semantic_types
        """ semantic type learned & suggested by Karma """
        self.learned_semantic_types: List[KarmaSemanticType] = learned_semantic_types
        self.literal_type: Optional[str] = literal_type
        self.is_literal_node: bool = is_literal_node

    @staticmethod
    def node2meta(self: 'KarmaGraphNode'):
        return {
            "user_semantic_types": [st for st in self.user_semantic_types],
            "learned_semantic_types": [st for st in self.learned_semantic_types],
            "literal_type": self.literal_type,
            "is_literal_node": self.is_literal_node
        }

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return {
            "user_semantic_types": [KarmaSemanticType.from_dict(st) for st in obj["user_semantic_types"]],
            "learned_semantic_types": [KarmaSemanticType.from_dict(st) for st in obj["learned_semantic_types"]],
            "literal_type": obj["literal_type"],
            "is_literal_node": obj["is_literal_node"]
        }

    @staticmethod
    def from_karma_model(node: dict, ont: Ontology,
                         id2columns: Dict[str, 'KarmaSourceColumn']) -> Tuple['KarmaGraphNode', int, bytes]:
        assert node['type'] in {'ColumnNode', 'InternalNode', 'LiteralNode'}, "Not recognized type: %s" % node['type']
        if node['type'] in {'ColumnNode', 'LiteralNode'}:
            type = GraphNodeType.DATA_NODE
        else:
            assert node['type'] == "InternalNode", node['type']
            type = GraphNodeType.CLASS_NODE
        is_literal_node = False

        if type == GraphNodeType.DATA_NODE:
            # IMPORTANT: this related to the SourceColumn::get_unique_column_name
            if node['type'] == 'LiteralNode':
                # trying to make short & readable label using heuristic
                label = node['value']
                is_literal_node = True
            else:
                label = id2columns[node['id']].column_name
        else:
            label = ont.simplify_uri(node['label']['uri'])

        user_semantic_types = []
        if 'user_semantic_types' in node:
            for x in node['user_semantic_types']:
                x = _dict_camel_to_snake(x)
                x['domain'] = ont.simplify_uri(x['domain']['uri'])
                x['type'] = ont.simplify_uri(x['type']['uri'])
                user_semantic_types.append(KarmaSemanticType(**x))

        # because there is duplication in data sources, now we filter out duplicated semantic types
        user_semantic_types = unique_values(user_semantic_types, key=lambda n: n.get_hashing_id())

        learned_semantic_types = []
        if 'learned_semantic_types' in node and node['learned_semantic_types'] is not None:
            for x in node['learned_semantic_types']:
                x = _dict_camel_to_snake(x)
                x['domain'] = ont.simplify_uri(x['domain']['uri'])
                x['type'] = ont.simplify_uri(x['type']['uri'])
                x['h_node_id'] = node['h_node_id']
                learned_semantic_types.append(KarmaSemanticType(**x))

        # double check data
        assert node['model_ids'] is None
        if 'rdf_literal_type' not in node or node['rdf_literal_type'] is None:
            literal_type = None
        else:
            literal_type = node['rdf_literal_type']['uri']
        return KarmaGraphNode(user_semantic_types, learned_semantic_types, literal_type,
                              is_literal_node), type, label.encode('utf-8')
