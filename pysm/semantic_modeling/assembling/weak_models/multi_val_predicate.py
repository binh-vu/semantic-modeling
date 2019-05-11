#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Dict, Tuple, List, Union, Optional, Set

from data_structure import Graph
from semantic_modeling.karma.semantic_model import SemanticModel


class MultiValuePredicate(object):
    instance = None

    def __init__(self, graphs: List[Graph], graph_ids: Set[str]) -> None:
        # probability of a predicate is a single value over list
        self.graph_ids = graph_ids
        self.prob: Dict[bytes, float] = {}

        link_usages: Dict[bytes, List[bool]] = defaultdict(lambda: [])
        for graph in graphs:
            for node in graph.iter_nodes():
                link_count: Dict[bytes, int] = defaultdict(lambda: 0)
                for link in node.iter_outgoing_links():
                    link_count[link.label] += 1

                for link_label, count in link_count.items():
                    link_usages[link_label].append(count == 1)

        for link_label, examples in link_usages.items():
            self.prob[link_label] = sum(examples) / len(examples)

    @staticmethod
    def get_instance(train_sms: List[SemanticModel]) -> 'MultiValuePredicate':
        sm_ids = {sm.id for sm in train_sms}

        if MultiValuePredicate.instance is None:
            MultiValuePredicate.instance = MultiValuePredicate([sm.graph for sm in train_sms], sm_ids)
            return MultiValuePredicate.instance

        assert sm_ids == MultiValuePredicate.instance.graph_ids
        return MultiValuePredicate.instance

    def compute_prob(self, link_label: bytes, link_no: int) -> Optional[float]:
        """How likely this link should be multi-value link"""
        if link_label not in self.prob:
            return None

        if link_no == 1:
            return None

        assert link_no > 1
        return 1 - self.prob[link_label]