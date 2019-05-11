#!/usr/bin/python
# -*- coding: utf-8 -*-
import copy
from collections import OrderedDict, defaultdict
from typing import Dict, Tuple, List, Set, Union, Optional, Any

import enum

from data_structure import Graph, GraphLink, GraphNode, GraphNodeType, GraphLinkType
from semantic_modeling.assembling.ont_graph import OntGraph, get_ont_graph
from semantic_modeling.data_io import get_ontology, get_semantic_models
from semantic_modeling.karma.karma_node import KarmaGraphNode, KarmaSemanticType
from semantic_modeling.karma.semantic_model import SemanticModel, Attribute
from semantic_modeling.utilities.ontology import Ontology


class Tag:
    ONT_GRAPH_SOURCE = 'ont_graph'
    NEW_DATA_SOURCE_TAG = 'new_data_source'


class IntGraphNode(GraphNode):

    def __init__(self, readable_id: str, tags: Set[str]) -> None:
        self.readable_id = readable_id
        self.tags = tags

    def clone(self):
        return IntGraphNode(self.readable_id, copy.copy(self.tags))


class IntGraphLink(GraphLink):

    def __init__(self, tags: Set[str]) -> None:
        self.tags = tags

    def clone(self):
        return IntGraphLink(copy.copy(self.tags))


class IntGraph(Graph):
    pass


def add_known_models(graph: IntGraph, train_sms: List[SemanticModel]) -> None:
    H: Dict[GraphNode, IntGraphNode] = {}
    for sm in train_sms:
        vertices: Dict[bytes, List[GraphNode]] = OrderedDict()
        for v in sm.graph.iter_class_nodes():
            if v.label not in vertices:
                vertices[v.label] = []
            vertices[v.label].append(v)

        for label, label_group in vertices.items():
            c1 = len(label_group)
            c2 = sum(1 for __ in graph.iter_nodes_by_label(label))

            for i in range(c2, c1):
                new_node = IntGraphNode(label.decode("utf-8") + str(i), set())
                graph.real_add_new_node(new_node, GraphNodeType.CLASS_NODE, label)

            # step 1, match sibling nodes first
            matched_nodes = list(graph.iter_nodes_by_label(label))
            for v in label_group:
                parent_labels = {l.get_source_node().label for l in v.iter_incoming_links()}
                unmapped_nodes = [n for n in matched_nodes if sm.id not in n.tags]
                siblings_unmapped_nodes = [
                    n for n in unmapped_nodes
                    if len(
                        parent_labels.intersection({l.get_source_node().label
                                                    for l in n.iter_incoming_links()})) > 0
                ]
                if len(siblings_unmapped_nodes) > 0:
                    v_prime = max(siblings_unmapped_nodes, key=lambda n: len(n.tags))
                    H[v] = v_prime
                    v_prime.tags.add(sm.id)

            # step 2, match the rest
            for v in label_group:
                if v in H:
                    continue
                unmapped_nodes = (n for n in matched_nodes if sm.id not in n.tags)
                v_prime = max(unmapped_nodes, key=lambda n: len(n.tags))
                H[v] = v_prime
                v_prime.tags.add(sm.id)

        for e in sm.graph.iter_links():
            u: GraphNode = e.get_source_node()
            v: GraphNode = e.get_target_node()

            if not (u.is_class_node() and v.is_data_node()):
                continue

            u_prime: IntGraphNode = H[u]
            for link in u_prime.iter_outgoing_links():
                if link.label == e.label:
                    v_prime: IntGraphNode = link.get_target_node()
                    break
            else:
                v_prime: IntGraphNode = IntGraphNode(v.label.decode('utf-8') + str(sum(1 for __ in graph.iter_nodes_by_label(v.label))), set())
                graph.real_add_new_node(v_prime, GraphNodeType.DATA_NODE, v.label)
            H[v] = v_prime
            v_prime.tags.add(sm.id)

        for e in sm.graph.iter_links():
            u: KarmaGraphNode = e.get_source_node()
            v: KarmaGraphNode = e.get_target_node()
            u_prime = H[u]
            v_prime = H[v]
            e_prime = next((ep for ep in v_prime.iter_incoming_links() if ep.source_id == u_prime.id and ep.label == e.label), None)

            if e_prime is None:
                e_prime = IntGraphLink(set())
                graph.real_add_new_link(e_prime, GraphLinkType.UNSPECIFIED, e.label, u_prime.id, v_prime.id)
            e_prime.tags.add(sm.id)


def add_semantic_types(graph: IntGraph, attributes: List[Attribute]):
    for attr in attributes:
        for st in attr.semantic_types:
            st_domain = st.domain.encode("utf-8")
            st_type = st.type.encode('utf-8')
            matched_nodes = list(graph.iter_nodes_by_label(st_domain))
            if len(matched_nodes) == 0:
                node = IntGraphNode({Tag.NEW_DATA_SOURCE_TAG})
                graph.real_add_new_node(node, GraphNodeType.CLASS_NODE, st_domain)
                matched_nodes = [node]

            for v in matched_nodes:
                if not any(e.label == st_type for e in v.iter_outgoing_links()):
                    w = IntGraphNode(attr.label, {Tag.NEW_DATA_SOURCE_TAG})
                    e = IntGraphLink({Tag.NEW_DATA_SOURCE_TAG})
                    graph.real_add_new_node(w, GraphNodeType.DATA_NODE, attr.label.encode('utf-8'))
                    graph.real_add_new_link(e, GraphLinkType.UNSPECIFIED, st_type, v.id, w.id)


def add_ont_paths(graph: IntGraph, ont: Ontology, ont_graph: OntGraph) -> None:
    for u in graph.iter_class_nodes():
        for v in graph.iter_class_nodes():
            if u == v:
                continue

            c1 = next(ont_graph.iter_nodes_by_label(u.label))
            c2 = next(ont_graph.iter_nodes_by_label(v.label))
            possible_predicates = ont_graph.get_possible_predicates(ont.full_uri(c1.label.decode('utf-8')), ont.full_uri(c2.label.decode('utf-8')))

            for p in possible_predicates:
                p_lbl = ont.simplify_uri(p.uri).encode('utf-8')
                e = next((e for e in v.iter_incoming_links() if e.source_id == u.id and e.label == p_lbl), None)
                if e is None:
                    e = IntGraphLink({Tag.ONT_GRAPH_SOURCE})
                    graph.real_add_new_link(e, GraphLinkType.UNSPECIFIED, p_lbl, u.id, v.id)

def create_psl_int_graph(train_sms: List[SemanticModel]):
    # TODO: Fix me!!, need to create proper psl int graph
    # quick solution is to create an int graph using mohsen method, and post processing to avoid multiple same structure
    # graphs

    g = IntGraph(True, True, True, 100, 100)
    add_known_models(g, train_sms)

    new_g = IntGraph(True, True, True, g.get_n_nodes(), g.get_n_links())
    idmaps = {}
    for n in g.iter_nodes():
        if n.is_class_node():
            idmaps[n.id] = new_g.real_add_new_node(n.clone(), n.type, n.label).id

    for n in g.iter_nodes():
        if n.is_class_node():
            targets = defaultdict(lambda: [])
            added_links = []
            for e in n.iter_outgoing_links():
                if e.get_target_node().is_data_node():
                    continue

                targets[e.target_id].append(e)

            for target_id, edges in targets.items():
                if len(edges) > 1:
                    target = g.get_node_by_id(target_id)
                    # will remove links if two substructures are same (now only handle leaf node case)
                    n_outgoing_links = sum(1 for e in target.iter_outgoing_links() if e.get_target_node().is_class_node())
                    if n_outgoing_links == 0:
                        # find another nodes that have same structures
                        for another_tid in targets:
                            if another_tid != target_id and g.get_node_by_id(another_tid).label == target.label:
                                if sum(1 for e in g.get_node_by_id(another_tid).iter_outgoing_links() if e.get_target_node().is_class_node()) == 0 and {e.label for e in targets[another_tid]} == {e.label for e in edges}:
                                    # they have same structures, now pick one
                                    assert len(edges) == 2
                                    targets[target_id] = [edges[0]]
                                    targets[another_tid] = [next(e for e in targets[another_tid] if e.label != edges[0].label)]

            for target_id, edges in targets.items():
                added_links += edges

            for e in added_links:
                new_g.real_add_new_link(e.clone(), e.type, e.label, idmaps[e.source_id], idmaps[e.target_id])

    return new_g


if __name__ == '__main__':
    dataset = "museum_edm"
    train_size = 6
    ont = get_ontology(dataset)
    ont_graph = get_ont_graph(dataset)
    semantic_models = get_semantic_models(dataset)

    g = IntGraph(True, True, True, 100, 100)
    add_known_models(g, semantic_models[:train_size])
    g.render(80)