from typing import List, Dict

from data_structure.graph_c.graph import GraphLink, GraphNode, Graph
from semantic_modeling.karma.karma import KarmaModel
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.utilities.ontology import Ontology


class SemanticType(object):
    domain: str
    type: str
    confidence_score: float

    def __init__(self, domain: str, type: str, confidence_score: float) -> None: ...

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(obj: dict) -> SemanticType: ...


class Attribute(object):
    id: int
    label: str
    semantic_types: List[SemanticType]

    def __init__(self, id: int, label: str, semantic_types: List[SemanticType]) -> None: ...

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(obj: dict) -> Attribute: ...

class SemanticModel(object):
    id: str
    attrs: List[Attribute]
    label2attrs: Dict[str, Attribute]
    graph: Graph

    def __init__(self, id: str, attrs: List[Attribute], graph: Graph, disable_check: bool=False) -> None: ...

    def get_attr_by_label(self, name: str) -> Attribute: ...

    def has_attr(self, name: str) -> bool: ...

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(obj: dict) -> SemanticModel: ...

    def to_karma_json_model(self, ont: Ontology) -> dict: ...

