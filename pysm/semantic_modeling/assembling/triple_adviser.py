#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Callable

from semantic_modeling.assembling.ont_graph import OntGraph
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.utilities.ontology import Ontology


class TripleAdviser(object):
    def get_pred_objs(self, subject: bytes) -> List[Tuple[bytes, bytes]]:
        """Get all possible (predicate, object)s that a subject can have"""
        pass

    def get_subj_preds(self, object: bytes) -> List[Tuple[bytes, bytes]]:
        """Get all possible (subject, predicate)s that a object can belong to"""
        pass


class OntologyTripleAdviser(TripleAdviser):
    def __init__(self, ont_graph: OntGraph, ont: Ontology) -> None:
        self.ont_graph: OntGraph = ont_graph
        self.ont = ont
        # we still keep str for things that interact with ontology since it invoke lots of string related operator
        # and unicode make the life much easier
        self.class_nodes: List[str] = ont_graph.get_potential_class_node_uris()
        self.data_nodes: Dict[bytes, List[KarmaSemanticType]] = {}

    def add_data_node(self, id: bytes, semantic_types: List[KarmaSemanticType]) -> None:
        self.data_nodes[id] = semantic_types

    def get_pred_objs(self, subject: bytes) -> List[Tuple[bytes, bytes]]:
        if subject in self.data_nodes:
            # this is data node, no outgoing link
            return []

        subject = self.ont.full_uri(subject.decode('utf-8'))
        doubles = []
        for object in self.class_nodes:
            for predicate in self.ont_graph.get_possible_predicates(subject, object):
                doubles.append((self.ont.simplify_uri(predicate.uri).encode('utf-8'),
                                self.ont.simplify_uri(object).encode('utf-8')))
        return doubles

    def get_subj_preds(self, object: bytes) -> List[Tuple[bytes, bytes]]:
        if object in self.data_nodes:
            # this is data node, use pre-defined semantic types
            return [(st.domain.encode('utf-8'), st.type.encode('utf-8')) for st in self.data_nodes[object]]

        object = self.ont.full_uri(object.decode('utf-8'))
        doubles = []
        for subject in self.class_nodes:
            for predicate in self.ont_graph.get_possible_predicates(subject, object):
                doubles.append((self.ont.simplify_uri(subject).encode('utf-8'),
                                self.ont.simplify_uri(predicate.uri).encode('utf-8')))
        return doubles


class EmpiricalTripleAdviser(OntologyTripleAdviser):
    def __init__(self,
                 ont_graph: OntGraph,
                 ont: Ontology,
                 prob_triple: Callable[[bytes, bytes, bytes], float],
                 max_candidates: int) -> None:
        super().__init__(ont_graph, ont)
        self.max_candidates: int = max_candidates
        self.prob_triple = prob_triple
        self.cached_subj_preds: Dict[bytes, List[Tuple[bytes, bytes]]] = {}
        self.cached_pred_objs: Dict[bytes, List[Tuple[bytes, bytes]]] = {}

    # @profile
    def get_subj_preds(self, object: bytes) -> List[Tuple[bytes, bytes]]:
        if object in self.data_nodes:
            # this is data node, use pre-defined semantic types
            return [(st.domain.encode('utf-8'), st.type.encode('utf-8')) for st in self.data_nodes[object]]

        if object not in self.cached_subj_preds:
            doubles = []
            for subj, pred in super().get_subj_preds(object):
                doubles.append(((subj, pred), self.prob_triple(subj, pred, object)))
            doubles = sorted(
                filter(lambda n: n[1] > 1e-9, doubles), key=lambda n: n[1], reverse=True)[:self.max_candidates]
            self.cached_subj_preds[object] = [x[0] for x in doubles]

        return self.cached_subj_preds[object]

    def get_pred_objs(self, subject: bytes) -> List[Tuple[bytes, bytes]]:
        if subject in self.data_nodes:
            # this is data node, doesn't have outgoing links
            return []

        if subject not in self.cached_pred_objs:
            doubles = []
            for pred, obj in super().get_pred_objs(subject):
                doubles.append(((pred, obj), self.prob_triple(subject, pred, obj)))
            doubles = sorted(
                filter(lambda n: n[1] > 1e-9, doubles), key=lambda n: n[1], reverse=True)[:self.max_candidates]
            self.cached_pred_objs[subject] = [x[0] for x in doubles]

        return self.cached_pred_objs[subject]
