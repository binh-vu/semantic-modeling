from collections import defaultdict
from typing import *

from data_structure import Graph
from karma_cli.app_misc import CliGraph, CliNode, NodeIdStr
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.karma.semantic_model import SemanticModel


class LinkSuggestion:

    def __init__(self, train_sms: List[SemanticModel]):
        self.train_sms = train_sms
        self.incoming_links: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(lambda: defaultdict(lambda: 0))

        for sm in self.train_sms:
            for n in sm.graph.iter_class_nodes():
                for e in n.iter_incoming_links():
                    self.incoming_links[n.label.decode()][(e.get_source_node().label.decode(), e.label.decode())] += 1

    def suggest_incoming_link(self, node_id: str, cli_graph: CliGraph) -> List[Tuple[NodeIdStr, str]]:
        node = cli_graph.nodes[node_id]
        raw_suggestion = sorted(self.incoming_links[node.label].keys(), key=lambda x: self.incoming_links[node.label][x], reverse=True)
        suggestion = []

        for node_class, predicate in raw_suggestion:
            count = 1
            for sim_node in cli_graph.iter_nodes_by_label(node_class):
                suggestion.append((NodeIdStr(sim_node.id), predicate))
                count += 1
            suggestion.append((NodeIdStr(f"{node_class}{count} (add)"), predicate))
        return suggestion


