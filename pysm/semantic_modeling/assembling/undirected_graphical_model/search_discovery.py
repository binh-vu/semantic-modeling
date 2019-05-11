#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import combinations
from pvectorc import pvector
from typing import Dict, List, Union, Optional, Callable, Iterable

from pyrsistent import pmap
from pyutils.list_utils import unique_values, _

from data_structure import Graph, GraphNodeType, GraphNode
from data_structure.graph_c.graph_util import to_hashable_string
from semantic_modeling.assembling.constructing_semantic_model import BeamSearchArgs, SearchNode, Tracker, \
    SearchNodeValue
from semantic_modeling.assembling.cshare.graph_explorer import GraphExplorer
from semantic_modeling.assembling.cshare.merge_graph import MergeGraph
from semantic_modeling.assembling.cshare.merge_planning import make_merge_plans, py_make_plan4case23, py_make_plan4case1
from semantic_modeling.assembling.graph_explorer_builder import GraphExplorerBuilder
from semantic_modeling.assembling.undirected_graphical_model.templates.substructure_template import \
    SubstructureFactorTemplate
from semantic_modeling.config import get_logger, config
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.karma.semantic_model import Attribute
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.serializable import serialize, serializeJSON


class PGMBeamSearchArgs(BeamSearchArgs):
    def __init__(self, source_id: str,
                 discovering_func: Callable[[List[SearchNode], BeamSearchArgs], List[SearchNode]], tracker: Tracker,
                 predict_graph_prob_func: Callable[[List[Graph]], List[float]],
                 graph_explorer_builder: GraphExplorerBuilder,
                 early_terminate_func: Optional[Callable[[int, Iterable[SearchNode]], bool]], beam_width: int,
                 gold_sm: Optional[Graph],
                 source_attributes: List[Attribute],
                 pre_filter_func: Optional[Callable[[Graph], bool]]=None) -> None:
        super().__init__(source_id, len(source_attributes), discovering_func, tracker, early_terminate_func)
        self.graph_explorer_builder: GraphExplorerBuilder = graph_explorer_builder
        self.incremental_id: int = 0
        self.beam_width = beam_width
        self.predict_graph_prob_func = predict_graph_prob_func
        self.gold_sm = gold_sm
        # use to select attributes to pick first
        self.top_attributes = sorted(
            [(attr.label.encode("utf-8"), attr.semantic_types[0].confidence_score) for attr in source_attributes
             if len(attr.semantic_types) > 0],
            key=lambda x: x[1],
            reverse=True)[:beam_width]
        self.top_attributes = [x[0] for x in self.top_attributes]
        self.pre_filter_func = pre_filter_func

    def get_and_increment_id(self):
        self.incremental_id += 1
        return self.incremental_id - 1


class PGMSearchNodeValue(SearchNodeValue):
    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self.graph: Graph = graph

    def get_hashing_id(self):
        """Create a hashing for this graph, but it only valid if this graph is a "tree" (only have one root)"""
        return to_hashable_string(self.graph)


class PGMStartSearchNode(SearchNode):
    def __init__(self, incremental_id: int, beam_search_args: PGMBeamSearchArgs,
                 remained_terminals: List[bytes]) -> None:

        super().__init__()
        self.incremental_id: int = incremental_id
        self.remained_terminals: List[bytes] = remained_terminals
        self.beam_search_args: PGMBeamSearchArgs = beam_search_args

        self.hashing_id = b"remained_terminals:%b" % b",".join(self.remained_terminals)

    def is_terminal(self, args: 'BeamSearchArgs') -> bool:
        return False

    def has_value(self) -> bool:
        return False

    def get_value(self) -> PGMSearchNodeValue:
        raise Exception("Start node doesn't have any value")

    def get_score(self) -> float:
        return float('-inf')

    def get_hashing_id(self) -> bytes:
        return self.hashing_id


class PGMSearchNode(SearchNode):
    def __init__(self, incremental_id: int, beam_search_args: PGMBeamSearchArgs, working_terminal: Optional[bytes],
                 remained_terminals: List[bytes], G_explorers: Dict[bytes, GraphExplorer],
                 G_terminals: Dict[bytes, Graph], G_scored: Dict[bytes, float]) -> None:

        super().__init__()
        self.incremental_id: int = incremental_id
        self.working_terminal: Optional[bytes] = working_terminal
        self.remained_terminals: List[bytes] = remained_terminals
        self.G_explorers: Dict[bytes, GraphExplorer] = G_explorers
        self.G_terminals: Dict[bytes, Graph] = G_terminals
        self.G_scored: Dict[bytes, float] = G_scored
        self.beam_search_args: PGMBeamSearchArgs = beam_search_args

        self.value: Optional[PGMSearchNodeValue] = PGMSearchNodeValue(
            self.G_terminals[self.working_terminal]) if self.working_terminal is not None else None

        # don't want to have duplicate value (we may have different working terminals, but same tree)
        self.hashing_id = self.value.get_hashing_id()
        # self.hashing_id = b"working_terminal:%b;remained_terminals:%b;G_terminal=%b" % (
        #     self.working_terminal, b",".join(self.remained_terminals), self.value.get_hashing_id()
        #     if self.value is not None else b"")

    def is_terminal(self, args: BeamSearchArgs) -> bool:
        return len(self.remained_terminals) == 0

    def get_value(self) -> PGMSearchNodeValue:
        return self.value

    def get_score(self) -> float:
        return self.G_scored[self.working_terminal]

    def get_hashing_id(self) -> bytes:
        return self.hashing_id


_logger = get_logger('app.assembling.search_discovery')


def filter_unlikely_graph(g: MergeGraph) -> bool:
    settings = Settings.get_instance()
    max_n_duplications = settings.mrf_max_n_duplications
    max_n_duplication_types = settings.mrf_max_n_duplication_types

    for n in g.iter_class_nodes():
        # FILTER middle nodes
        if n.n_incoming_links == 1 and n.n_outgoing_links == 1:
            link = next(iter(n.iter_outgoing_links()))
            if link.get_target_node().is_class_node():
                return False

        # FILTER: max_size_duplication_group <= 7 and max_n_duplications <= 4
        n_duplication_types = 0
        for e_lbl, es in _(n.iter_outgoing_links()).imap(lambda e: (e.label, e)).group_by_key().get_value():
            if len(es) > max_n_duplications:
                return False

            if len(es) > 1:
                n_duplication_types += 1

        if n_duplication_types > max_n_duplication_types:
            return False

    return True


# @profile
def discovering_func(search_nodes: List[Union[PGMStartSearchNode, PGMSearchNode]],
                     args: PGMBeamSearchArgs) -> List[PGMSearchNode]:
    global _logger
    next_nodes: List[PGMSearchNode] = []
    merged_plans = []

    if isinstance(search_nodes[0], PGMStartSearchNode):
        # can only have one starter node
        search_node = search_nodes[0]
        G_explorers: Dict[bytes, GraphExplorer] = {}
        G_terminals: Dict[bytes, Graph] = {}
        G_scored: Dict[bytes, float] = {}

        # create graph & graph explorer for each terminals
        for terminal in search_node.remained_terminals:
            g: Graph = Graph(index_node_type=True, index_node_label=True)
            g.add_new_node(GraphNodeType.DATA_NODE, terminal)

            G_terminals[terminal] = g
            G_scored[terminal] = 1
            G_explorers[terminal] = args.graph_explorer_builder.build(g)

        search_node.G_terminals = pmap(G_terminals)
        search_node.G_scored = pmap(G_scored)
        search_node.G_explorers = pmap(G_explorers)

        search_node.remained_terminals = pvector(search_node.remained_terminals)

        # final all possible merged points between every terminal pairs & release it as terminal nodes
        # TOO EXPENSIVE
        # for T_i, T_j in (
        #         tuple(c)
        #         for c in unique_values(frozenset(c) if c[0] != c[1] else c for c in combinations(search_node.remained_terminals, 2))):
        #     G_ti, G_tj = G_terminals[T_i], G_terminals[T_j]
        for T_i in args.top_attributes:
            for T_j in search_node.remained_terminals:
                if T_i == T_j:
                    continue
                G_ti, G_tj = G_terminals[T_i], G_terminals[T_j]

                merged_plans += [(T_i, T_j, plan, search_node,
                                  MergeGraph.create(G_ti, G_tj, plan.int_tree, plan.int_a, plan.int_b))
                                 for plan in py_make_plan4case1(G_ti, G_tj, G_explorers[T_i], G_explorers[T_j])]

        # doing filter to speed up, will remove all merge graph that have more than 3 nodes (because the good result is usually
        # two data nodes connect to one single class node)
        merged_plans = [x for x in merged_plans if x[-1].get_n_nodes() == 3]
    else:
        for search_node in search_nodes:
            T_i = search_node.working_terminal
            G_ti_explorer = search_node.G_explorers[T_i]
            G_ti = search_node.G_terminals[T_i]
            for T_j in unique_values(search_node.remained_terminals):
                G_tj = search_node.G_terminals[T_j]
                merged_plans += [(T_i, T_j, plan, search_node,
                                  MergeGraph.create(G_ti, G_tj, plan.int_tree, plan.int_a, plan.int_b))
                                 for plan in make_merge_plans(G_ti, G_tj, G_ti_explorer, search_node.G_explorers[T_j])]

    if args.pre_filter_func is not None:
        n_next_states = len(merged_plans)
        filtered_merged_plans = []
        for merged_plan in merged_plans:
            if args.pre_filter_func(merged_plan[-1]):
                filtered_merged_plans.append(merged_plan)

        merged_plans = filtered_merged_plans
        _logger.debug("(%s) #possible next states: %s (filtered down to: %s)", args.source_id, n_next_states, len(merged_plans))
    else:
        _logger.debug("(%s) #possible next states: %s", args.source_id, len(merged_plans))

    merged_graphs = [x[-1] for x in merged_plans]
    merged_probs = args.predict_graph_prob_func(merged_graphs)

    best_plans = sorted(
        zip(merged_plans, merged_graphs, merged_probs), key=lambda x: x[-1], reverse=True)[:args.beam_width]

    need_remove_T_i: bool = isinstance(search_nodes[0], PGMStartSearchNode)

    for merged_plan, merged_graph, score, in best_plans:
        T_i, T_j, __, search_node, __ = merged_plan
        working_terminal = b'%b---%b' % (T_i, T_j)
        remained_terminals = search_node.remained_terminals.remove(T_j)
        if need_remove_T_i:
            remained_terminals = remained_terminals.remove(T_i)

        g: Graph = merged_graph.proceed_merging()
        current_G_explorers = search_node.G_explorers.set(working_terminal, args.graph_explorer_builder.build(g))
        current_G_terminals = search_node.G_terminals.set(working_terminal, g)
        current_G_scored = search_node.G_scored.set(working_terminal, score)
        next_nodes.append(
            PGMSearchNode(args.get_and_increment_id(), args, working_terminal, remained_terminals, current_G_explorers,
                          current_G_terminals, current_G_scored))

    return next_nodes
