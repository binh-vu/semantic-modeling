#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Dict, List, Callable, Iterable

import sys

from semantic_modeling.config import get_logger
from semantic_modeling.utilities.serializable import serialize


class SearchNodeValue(object):
    def get_hashing_id(self) -> str:
        pass


class SearchNode(object):
    def get_hashing_id(self) -> str:
        pass

    def get_value(self) -> SearchNodeValue:
        pass

    def has_value(self) -> bool:
        # for start node, some node like start doesn't have any value
        return True

    def get_score(self) -> float:
        pass

    def is_terminal(self, args: 'BeamSearchArgs') -> bool:
        pass


class Tracker(object):
    def __init__(self, track_search_nodes: bool = False) -> None:
        self.track_search_nodes = track_search_nodes
        self.list_search_nodes: List[List[SearchNode]] = []

    def log_search_nodes(self, search_nodes: Iterable[SearchNode]):
        if self.track_search_nodes:
            self.list_search_nodes.append(list(search_nodes))

    def serialize(self, output: str):
        serialize({
            'list_search_nodes': [[(n.get_value(), n.get_score()) for n in search_nodes]
                                  for search_nodes in self.list_search_nodes]
        }, output)


class BeamSearchArgs(object):
    def __init__(self,
                 source_id: str,
                 n_attrs: int,
                 discovering_func: Callable[[List[SearchNode], 'BeamSearchArgs'], List[SearchNode]],
                 tracker: Tracker,
                 early_terminate_func: Callable[[int, Iterable[SearchNode]], bool] = None) -> None:
        self.source_id = source_id
        self.n_attrs = n_attrs
        self.discovering_func = discovering_func
        self.tracker: Tracker = tracker
        self.early_terminate_func: Callable[[int, Iterable[SearchNode]], bool] = early_terminate_func

    def should_stop(self, n_iter: int, current_nodes: Iterable[SearchNode]) -> bool:
        if self.early_terminate_func is None:
            return False

        return self.early_terminate_func(n_iter, current_nodes)


_logger = get_logger('app.assembling.beam_search')


# @profile
def beam_search(starts: List[SearchNode], beam_width: int, n_results: int, args: BeamSearchArgs) -> List[SearchNode]:
    global _logger

    assert beam_width >= len(starts)
    # store the search result, a map from id of node's value => node to eliminate duplicated result
    results: Dict[str, SearchNode] = {}

    # ##############################################
    # Add very first nodes to kick off BEAM SEARCH
    current_exploring_nodes: Dict[str, SearchNode] = OrderedDict()
    for n in starts:
        current_exploring_nodes[n.get_hashing_id()] = n

    # ##############################################
    # START BEAM SEARCH!!!
    prev_exploring_nodes: Dict[str, SearchNode] = None
    n_iter = 0

    while len(results) < n_results and len(current_exploring_nodes) > 0:
        _logger.debug("=== (%s) BEAMSEARCH ===> Doing round: %s (n_attrs=%d)", args.source_id, n_iter + 1, args.n_attrs)

        next_exploring_nodes: List[SearchNode] = args.discovering_func(list(current_exploring_nodes.values()), args)
        # for exploring_node in current_exploring_nodes.values():
        #     for next_exploring_node in args.discovering_func(exploring_node, args):
        #         next_exploring_nodes.append(next_exploring_node)

        # sort next nodes by their score, higher is better
        next_exploring_nodes.sort(key=lambda n: n.get_score(), reverse=True)
        prev_exploring_nodes = current_exploring_nodes
        current_exploring_nodes: Dict[str, SearchNode] = OrderedDict()

        for i, next_node in enumerate(next_exploring_nodes):
            if next_node.is_terminal(args):
                results[next_node.get_value().get_hashing_id()] = next_node
                if len(results) == n_results:
                    break
            else:
                current_exploring_nodes[next_node.get_hashing_id()] = next_node
                if len(current_exploring_nodes) == beam_width:
                    break

        n_iter += 1
        if len(results) == n_results:
            break

        if len(current_exploring_nodes) > 0:
            args.tracker.log_search_nodes(current_exploring_nodes.values())

        # TODO: self-pruning to see if we can improve the result
        if args.should_stop(n_iter, current_exploring_nodes.values()):
            break

    # ##############################################
    # Add more results to fulfill the requirements
    if len(results) == 0:
        if len(current_exploring_nodes) == 0:
            exploring_nodes = prev_exploring_nodes
        else:
            exploring_nodes = current_exploring_nodes

        if exploring_nodes is not None:
            for exploring_node in exploring_nodes.values():
                if len(results) < n_results and exploring_node.has_value():
                    results[exploring_node.get_value().get_hashing_id()] = exploring_node
                else:
                    break

    return sorted(results.values(), key=lambda n: n.get_score(), reverse=True)
