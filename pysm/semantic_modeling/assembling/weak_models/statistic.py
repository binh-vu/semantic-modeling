#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional

from data_structure import Graph
from semantic_modeling.karma.semantic_model import SemanticModel


class Statistic(object):

    instance = None

    def __init__(self, graphs: List[Graph], graph_ids: Set[str]) -> None:
        """provide lots of prob. estimated by counting"""
        self.graph_ids = graph_ids

        # count of link given source & object
        self.c_l_given_so: Dict[Tuple[bytes, bytes], Dict[bytes, int]] = {}
        # count of nodes
        self.c_n: Dict[bytes, int] = {}
        # count of link given source
        self.c_l_given_s: Dict[bytes, Dict[bytes, int]] = {}

        # COMPUTE counting
        for g in graphs:
            for link in g.iter_links():
                s = link.get_source_node().label
                o = link.get_target_node().label

                # COMPUTE c_l_given_s
                if s not in self.c_l_given_s:
                    self.c_l_given_s[s] = {}
                if link.label not in self.c_l_given_s[s]:
                    self.c_l_given_s[s][link.label] = 0
                self.c_l_given_s[s][link.label] += 1

                # COMPUTE c_l_given_so
                if link.get_target_node().is_data_node():
                    # no need to estimate this prob, since it will be result from semantic labeling
                    pass
                else:
                    if (s, o) not in self.c_l_given_so:
                        self.c_l_given_so[(s, o)] = {}
                    if link.label not in self.c_l_given_so[(s, o)]:
                        self.c_l_given_so[(s, o)][link.label] = 0
                    self.c_l_given_so[(s, o)][link.label] += 1

            # COMPUTE c_n
            for n in g.iter_nodes():
                if n.label not in self.c_n:
                    self.c_n[n.label] = 0
                self.c_n[n.label] += 1

        # cached
        self.p_critical_l_given_s = {}
        for s, counts in self.c_l_given_s.items():
            l, c_l = max(counts.items(), key=lambda x: x[1])
            self.p_critical_l_given_s[s] = (l, c_l / self.c_n[s])

    @staticmethod
    def get_instance(semantic_mappings: List[SemanticModel]) -> 'Statistic':
        sm_ids = {sm.id for sm in semantic_mappings}

        if Statistic.instance is None:
            Statistic.instance = Statistic([sm.graph for sm in semantic_mappings], sm_ids)
            return Statistic.instance

        assert sm_ids == Statistic.instance.graph_ids
        return Statistic.instance

    def p_critial_link(self, s: bytes) -> Optional[Tuple[bytes, float]]:
        """Return a most important link and probability that its should be presented in all nodes"""
        if s not in self.c_l_given_s:
            return None
        return self.p_critical_l_given_s[s]

    def p_n(self, lbl: bytes, default: float):
        """Compute prior of a class label P(node=lbl)"""
        if lbl not in self.c_n:
            return default
        return self.c_n[lbl] / sum(self.c_n.values())

    def p_l_given_so(self, s: bytes, l: bytes, o: bytes, default: float):
        """Compute P(predicate=l|source=s,object=o)"""
        if (s, o) not in self.c_l_given_so:
            return default
        return self.c_l_given_so[(s, o)].get(l, 0) / sum(self.c_l_given_so[(s, o)].values())

    def p_triple(self, s: bytes, l: bytes, o: bytes, default: float=0.0):
        """Compute P(source=s, predicate=l, object=0)"""
        return self.p_n(s, default) * self.p_n(o, default) * self.p_l_given_so(s, l, o, default)
