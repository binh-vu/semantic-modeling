#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from typing import Dict, Tuple, List, Set, Union, Optional, TypeVar

from data_structure import GraphLink, GraphLinkType
from semantic_modeling.utilities.ontology import Ontology

T = TypeVar('T')


def _camel_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def _dict_camel_to_snake(mapping: Dict[str, T]) -> Dict[str, T]:
    return {_camel_to_snake(key): value for key, value in mapping.items()}


class KarmaGraphLink(GraphLink):
    def __init__(self, weight: float) -> None:
        self.weight = weight

    @staticmethod
    def link2meta(self: 'KarmaGraphLink'):
        return {"weight": self.weight}

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return {"weight": obj["weight"]}

    @staticmethod
    def from_karma_model(link: dict, ont: Ontology) -> Tuple['KarmaGraphLink', int, bytes, int, int]:
        if link['type'] == 'ObjectPropertyLink':
            link_type = GraphLinkType.OBJECT_PROPERTY
        elif link['type'] == 'DataPropertyLink':
            link_type = GraphLinkType.DATA_PROPERTY
        else:
            assert link['type'] == 'ClassInstanceLink'
            link_type = GraphLinkType.URI_PROPERTY

        return KarmaGraphLink(link['weight']), link_type, ont.simplify_uri(
            link['label']['uri']).encode('utf-8'), link['source_id'], link['target_id']
