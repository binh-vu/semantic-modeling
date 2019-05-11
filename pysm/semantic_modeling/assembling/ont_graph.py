#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

import ontospy
from nose.tools import eq_
from ontospy.core.entities import OntoClass

from data_structure import GraphLinkType, GraphNode, GraphNodeType, GraphLink, Graph, graph2dict, dict2graph
from semantic_modeling.config import config
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeJSON, deserialize, serializeJSON


class PredicateConstraint(Enum):
    NO_CONSTRAINT = 'NO_CONSTRAINT'
    WITH_ONLY_DOMAIN_CONSTRAINT = 'WITH_ONLY_DOMAIN_CONSTRAINT'
    WITH_ONLY_RANGE_CONSTRAINT = 'WITH_ONLY_RANGE_CONSTRAINT'
    WITH_DOMAIN_RANGE_CONSTRAINT = 'WITH_DOMAIN_RANGE_CONSTRAINT'


class PredicateType(Enum):
    OWL_DATA_PROP = 'owl:DatatypeProperty'
    OWL_OBJECT_PROP = 'owl:ObjectProperty'
    OWL_ANNOTATION_PROP = 'owl:AnnotationProperty'
    RDF_PROP = 'rdf:Property'


class Predicate(object):

    def __init__(
        self, uri: str, domains: List[str], ranges: List[str], rdf_type: str, is_rdf_type_reliable: bool,
        defined_in_onts: Set[str]
    ) -> None:
        self.uri = uri
        self.domains: Set[str] = set(domains)
        self.ranges: Set[str] = set(ranges)
        self.rdf_type: PredicateType = PredicateType(rdf_type)
        self.is_rdf_type_reliable: bool = is_rdf_type_reliable
        self.defined_in_onts: Set[str] = defined_in_onts

    def to_dict(self):
        return {
            "uri": self.uri,
            "domains": self.domains,
            "ranges": self.ranges,
            "rdf_type": self.rdf_type.value,
            "is_rdf_type_reliable": self.is_rdf_type_reliable,
            "defined_in_onts": self.defined_in_onts
        }

    @staticmethod
    def from_dict(obj: dict) -> 'Predicate':
        return Predicate(
            obj['uri'], obj['domains'], obj['ranges'], obj['rdf_type'], obj['is_rdf_type_reliable'],
            set(obj['defined_in_onts'])
        )

    def get_constraint(self):
        if len(self.domains) > 0 and len(self.ranges) > 0:
            return PredicateConstraint.WITH_DOMAIN_RANGE_CONSTRAINT
        elif len(self.domains) > 0:
            return PredicateConstraint.WITH_ONLY_DOMAIN_CONSTRAINT
        elif len(self.ranges) > 0:
            return PredicateConstraint.WITH_ONLY_RANGE_CONSTRAINT
        else:
            return PredicateConstraint.NO_CONSTRAINT

    def merge(self, o: 'Predicate') -> None:
        assert self.uri == o.uri
        self.domains = self.domains.union(o.domains)
        self.ranges = self.ranges.union(o.ranges)
        if o.rdf_type == 'rdf:Property':
            # just ignore it
            pass
        elif self.rdf_type is PredicateType.RDF_PROP:
            self.rdf_type = o.rdf_type
        elif self.rdf_type == PredicateType.OWL_ANNOTATION_PROP and o.rdf_type != PredicateType.OWL_ANNOTATION_PROP:
            self.rdf_type = o.rdf_type
        elif o.rdf_type == PredicateType.OWL_ANNOTATION_PROP:
            pass
        else:
            eq_(self.rdf_type, o.rdf_type)

        self.is_rdf_type_reliable = self.is_rdf_type_reliable or o.is_rdf_type_reliable
        self.defined_in_onts = self.defined_in_onts.union(o.defined_in_onts)

    def __eq__(self, o: object) -> bool:
        if o is None or not isinstance(o, Predicate):
            return False
        if self.uri != o.uri or self.domains != o.domains or self.ranges != o.ranges or self.rdf_type != o.rdf_type:
            return False

        return True

    def __repr__(self):
        return 'Predicate(uri=%s, rdf_type=%s, is_rdftype_reliable=%s, domains=%s, ranges=%s, defined_in_onts=%s)' % (
            self.uri, self.rdf_type, self.is_rdf_type_reliable, self.domains, self.ranges, self.defined_in_onts
        )


class OntGraphLink(GraphLink):

    def __init__(self, predicate: Predicate) -> None:
        self.predicate = predicate

    @staticmethod
    def link2meta(self):
        """For JSON serialization"""
        return {"predicate": self.predicate.to_dict()}

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return {"predicate": Predicate.from_dict(obj["predicate"])}


class OntGraphNode(GraphNode):

    def __init__(self, uri: str, parents_uris: Set[str], children_uris: Set[str]) -> None:
        self.uri = uri
        self.parents_uris: Set[str] = parents_uris
        self.children_uris: Set[str] = children_uris

    @staticmethod
    def node2meta(self):
        """For JSON serialization"""
        return {"uri": self.uri, "parents_uris": self.parents_uris, "children_uris": self.children_uris}

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return dict(uri=obj['uri'], parents_uris=set(obj['parents_uris']), children_uris=set(obj['children_uris']))

    def has_parent(self, node_uri: str) -> bool:
        if node_uri in self.parents_uris:
            return True

        graph: OntGraph = self._graph
        for parent_uri in self.parents_uris:
            if graph.get_node_by_uri(parent_uri).has_parent(node_uri):
                return True

        return False

    def has_child(self, node_uri: str) -> bool:
        if node_uri in self.children_uris:
            return True

        graph: OntGraph = self._graph
        for child_uri in self.children_uris:
            if graph.get_node_by_uri(child_uri).has_child(node_uri):
                return True

        return False

    def __repr__(self) -> str:
        return 'OntGraphNode(id=%s, type=%s, label=%s, uri=%s, parents_ids=%s, children_ids=%s)' % (
            self.id, self.type, self.label, self.uri, self.parents_uris, self.children_uris
        )


class OntGraph(Graph):

    def __init__(self, id: str, predicates: Optional[List[Predicate]] = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.id: str = id
        self.predicates: List[Predicate] = predicates or []
        self.uri_index: Dict[str, int] = {}

    @staticmethod
    def graph2meta(self):
        return {"id": self.id, "predicates": [p.to_dict() for p in self.predicates]}

    @staticmethod
    def meta2args(obj: dict) -> dict:
        return dict(id=obj['id'], predicates=[Predicate.from_dict(p) for p in obj['predicates']])

    def to_dict(self):
        return graph2dict(
            self, graph2meta=self.graph2meta, node2meta=OntGraphNode.node2meta, link2meta=OntGraphLink.link2meta
        )

    @staticmethod
    def from_dict(obj) -> 'OntGraph':
        return dict2graph(
            obj, OntGraph, OntGraphNode, OntGraphLink, OntGraph.meta2args, OntGraphNode.meta2args,
            OntGraphLink.meta2args
        )

    def real_add_new_node(self, node: OntGraphNode, type: GraphNodeType, label: bytes):
        super(OntGraph, self).real_add_new_node(node, type, label)
        self.uri_index[node.uri] = node.id
        return node

    def add_new_node(
        self, type: GraphNodeType, label: bytes, uri: str, parents_uris: Set[str], children_uris: Set[str]
    ):
        # this calling directly to super() api, so it doesn't touch local real_add_new_node
        node = super().real_add_new_node(OntGraphNode(uri, parents_uris, children_uris), type, label)
        self.uri_index[node.uri] = node.id
        return node

    def has_node_with_uri(self, uri: str) -> bool:
        return uri in self.uri_index

    def get_node_by_uri(self, uri: str) -> OntGraphNode:
        return self.get_node_by_id(self.uri_index[uri])

    def get_potential_class_node_uris(self) -> List[str]:
        """Get all labels that can be a class node in any source modeling by this ontology"""
        nodes = [n for n in self.iter_nodes() if n.is_class_node()]
        uris = []
        ignored_namespaces = [
            'http://www.w3.org/2000/01/rdf-schema#', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'http://www.w3.org/2001/XMLSchema#', 'http://www.w3.org/2002/07/owl#Class'
        ]
        for n in nodes:
            if any((n.uri.startswith(ignored_ns) for ignored_ns in ignored_namespaces)):
                continue
            uris.append(n.uri)
        return uris

    def set_predicates(self, predicates: List[Predicate]) -> 'OntGraph':
        self.predicates = predicates
        return self

    def get_possible_predicates(self, source_uri: str, target_uri: str) -> List[Predicate]:
        possible_predicates = []
        source = self.get_node_by_uri(source_uri)
        if source.type != GraphNodeType.CLASS_NODE:
            return []

        target = self.get_node_by_uri(target_uri)
        for predicate in self.predicates:
            if self.is_linkable(predicate, source, target):
                possible_predicates.append(predicate)
        return possible_predicates

    def is_linkable(self, predicate: Predicate, source: OntGraphNode, target: OntGraphNode) -> bool:
        if len(predicate.domains) > 0:
            if source.uri not in predicate.domains and len(predicate.domains.intersection(source.parents_uris)) == 0:
                return False

        if len(predicate.ranges) > 0:
            if target.uri not in predicate.ranges and len(predicate.ranges.intersection(target.parents_uris)) == 0:
                return False
        else:
            if predicate.is_rdf_type_reliable:
                if (
                    predicate.rdf_type == PredicateType.OWL_DATA_PROP or
                    predicate.rdf_type == PredicateType.OWL_ANNOTATION_PROP
                ) and target.type != GraphNodeType.DATA_NODE:
                    return False
                if predicate.rdf_type == PredicateType.OWL_OBJECT_PROP and target.type != GraphNodeType.CLASS_NODE:
                    return False

        return True

    def render2txt(self, f_output: str) -> None:
        super().render2txt(f_output)
        with open(f_output, 'a') as f:
            f.write('Predicates:\n')
            for p in self.predicates:
                f.write(
                    '    + uri: %s, rdf_type: %s, domain: %s, ranges: %s\n' % (p.uri, p.rdf_type, p.domains, p.ranges)
                )


def is_data_node(uri: str) -> bool:
    return uri in {'http://www.w3.org/2000/01/rdf-schema#Literal', 'http://www.w3.org/2001/XMLSchema#dateTime'}


def filter_uri(uri: str) -> bool:
    """Determine which uri could be used to create node"""
    return uri not in {"http://www.w3.org/2002/07/owl#Class"}


def create_node_args(ont: Ontology, cls: OntoClass) -> Optional[Tuple[int, bytes, str, Set[str], Set[str]]]:
    if not filter_uri(str(cls.uri)):
        return None
    if is_data_node(str(cls.uri)):
        node_type = GraphNodeType.DATA_NODE
    else:
        node_type = GraphNodeType.CLASS_NODE

    if cls.sparqlHelper is not None:
        parents = {str(x[0]) for x in cls.sparqlHelper.getClassAllSupers(cls.uri)}
    else:
        parents = set()
    children = set()
    return node_type, ont.simplify_uri(str(cls.uri)).encode('utf-8'), str(cls.uri), parents, children


def add_node(ont: Ontology, ont_graph: OntGraph, cls: OntoClass) -> None:
    node_args = create_node_args(ont, cls)
    if node_args is None:
        return

    if not ont_graph.has_node_with_uri(node_args[2]):
        ont_graph.add_new_node(*node_args)
        return

    previous_node: OntGraphNode = ont_graph.get_node_by_uri(node_args[2])
    previous_node.parents_uris = previous_node.parents_uris.union(node_args[3])
    previous_node.children_uris = previous_node.children_uris.union(node_args[4])


def build_ont_graph(dataset: str) -> OntGraph:
    ont = Ontology.from_dataset(dataset)
    ont_graph: OntGraph = OntGraph(dataset)
    predicates: Dict[str, Predicate] = {}

    for ont_name, ont_conf in config.datasets[dataset].ontology.items():
        fpaths = []
        if 'fpath' in ont_conf:
            fpaths = [ont_conf.fpath]
        elif 'fpaths' in ont_conf:
            fpaths = [ont_conf.fpaths]

        for fpath in fpaths:
            g = ontospy.Ontospy(str(fpath.as_path()))
            is_rdf_type_reliable = False

            for cls in g.classes:
                add_node(ont, ont_graph, cls)

            for prop in g.properties:
                for rg in prop.ranges:
                    add_node(ont, ont_graph, rg)
                for domain in prop.domains:
                    add_node(ont, ont_graph, domain)

                try:
                    predicate = Predicate(
                        str(prop.uri), [str(x.uri) for x in prop.domains], [str(x.uri) for x in prop.ranges],
                        ont.simplify_uri(str(prop.rdftype)), False, {ont_name}
                    )

                    if str(prop.uri) in predicates:
                        predicates[str(prop.uri)].merge(predicate)
                    else:
                        predicates[str(prop.uri)] = predicate

                    if predicate.rdf_type in {PredicateType.OWL_DATA_PROP, PredicateType.OWL_OBJECT_PROP}:
                        is_rdf_type_reliable = True
                except Exception:
                    print(ont_name, prop)
                    print(prop.__dict__)
                    raise

            for uri, predicate in predicates.items():
                if ont_name in predicate.defined_in_onts:
                    predicate.is_rdf_type_reliable = is_rdf_type_reliable

    ont_graph.set_predicates(list(predicates.values()))
    # update parent & children between nodes
    for node in ont_graph.iter_nodes():
        for node_uri in node.parents_uris.union(node.children_uris):
            if not ont_graph.has_node_with_uri(node_uri):
                # node is referred by subClassOf but never been defined before
                ont_graph.add_new_node(
                    GraphNodeType.CLASS_NODE,
                    ont.simplify_uri(node_uri).encode('utf-8'), node_uri, set(), set()
                )

    for node in ont_graph.iter_nodes():
        for parent_uri in node.parents_uris:
            ont_graph.get_node_by_uri(parent_uri).children_uris.add(node.uri)
        for child_uri in node.children_uris:
            ont_graph.get_node_by_uri(child_uri).parents_uris.add(node.uri)
    return ont_graph


_ont_graph_vars = {}


def get_ont_graph(dataset: str) -> OntGraph:
    global _ont_graph_vars

    if dataset not in _ont_graph_vars:
        # if it hasn't been cached
        cache_file = Path(config.fsys.debug.as_path() + f'/{dataset}/cached/ont_graph.json')
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        if cache_file.exists():
            ont_graph = deserializeJSON(cache_file, Class=OntGraph)
        else:
            ont_graph: OntGraph = build_ont_graph(dataset)
            serializeJSON(ont_graph, cache_file)

        _ont_graph_vars[dataset] = ont_graph

    return _ont_graph_vars[dataset]


if __name__ == '__main__':
    dataset = 'museum_crm'
    # import sys
    #
    # def trace(frame, event, arg):
    #     if frame.f_code.co_filename.startswith("/Users/rook/workspace/DataIntegration/SourceModeling/"):
    #         print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    #     else:
    #         print(".", end="")
    #     return trace
    #
    # args = {'children_uris': set(),
    #     'parents_uris': set(),
    #     'uri': 'http://www.w3.org/2000/01/rdf-schema#Resource'}
    # # OntGraphNode("haha", set(), set())
    # a = OntGraphNode(**args)
    # print("==========")
    # # print(a.uri)
    # # sys.settrace(trace)
    ont_graph: OntGraph = deserializeJSON(config.fsys.debug.as_path() + '/%s/cached/ont_graph.json' % dataset, OntGraph)
    ont: Ontology = deserialize(config.fsys.debug.as_path() + '/%s/cached/ont.pkl' % dataset)
    # print(a.uri)
    # print("========SIGSEGV IN DEBUG MODE==")

    # ont = Ontology.from_data_source(data_source)
    # ont_graph = build_ont_graph(data_source)
    #
    # # %%
    #
    # # ont_graph.render2txt(config.fsys.debug.as_path() + '/%s/ont_graph.txt' % data_source)
    #
    # # %%
    s1 = ont.full_uri('crm:E63_Beginning_of_Existence')
    s2 = ont.full_uri('crm:E52_Time-Span')
    for predicate in ont_graph.get_possible_predicates(s1, s2):
        # if link.label.find('aggregatedCHO') != -1:
        print(predicate)
        print('=' * 20)

    print('Total', len(ont_graph.get_possible_predicates(s1, s2)))
