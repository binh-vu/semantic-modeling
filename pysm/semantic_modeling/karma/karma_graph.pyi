from typing import *
from data_structure import Graph
from semantic_modeling.karma.karma import KarmaModel, KarmaSourceColumn
from semantic_modeling.karma.karma_link import KarmaGraphLink
from semantic_modeling.karma.karma_node import KarmaGraphNode
from semantic_modeling.utilities.ontology import Ontology


class KarmaGraph(Graph[KarmaGraphNode, KarmaGraphLink]):
    model: KarmaModel

    def __init__(self,
             index_node_type=True,
             index_node_label=True,
             index_link_label=True,
             estimated_n_nodes=24,
             estimated_n_links=23,
             name=b"graph") -> None: ...

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(obj: dict) -> KarmaGraph: ...

    def set_model(self, model: KarmaModel) -> KarmaGraph: ...

    def to_graph(self) -> Graph: ...

    @staticmethod
    def from_karma_model(name: bytes, graph: dict, ontology: Ontology,
                         id2columns: Dict[str, KarmaSourceColumn]) -> Tuple[KarmaGraph, dict, dict]: ...
