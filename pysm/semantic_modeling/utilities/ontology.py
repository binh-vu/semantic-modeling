#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import rdflib
from rdflib import RDF, OWL, RDFS
from rdflib.plugins.memory import IOMemory
from typing import List

from semantic_modeling.config import config


class Ontology(object):

    def __init__(self, id: str) -> None:
        self.store = IOMemory()
        self.g = rdflib.Graph(store=self.store)
        # default ontology
        self.namespaces = {
            'owl': 'http://www.w3.org/2002/07/owl#'
        }
        self.reversed_namespaces = {}
        self.id: str = id

    @staticmethod
    def from_dataset(dataset: str) -> 'Ontology':
        """Load ontologies that are used to build data source"""
        ont = Ontology(dataset)
        ont_conf = config.datasets[dataset].ontology
        for prefix in ont_conf:
            namespace = ont_conf[prefix].namespace
            if 'fpath' in ont_conf[prefix]:
                fpath = ont_conf[prefix].fpath.as_path()
                ont.load_ontology(str(fpath), str(prefix), str(namespace))
            elif 'fpaths' in ont_conf[prefix]:
                for fpath in ont_conf[prefix].fpaths:
                    ont.load_ontology(str(fpath.as_path()), str(prefix), str(namespace))
            else:
                ont.namespaces[str(prefix)] = str(namespace)
                ont.reversed_namespaces[str(namespace)] = str(prefix)
        return ont

    def load_ontology(self, fpath: str, prefix: str=None, namespace: str=None) -> 'Ontology':
        """
            :param fpath: file name of the ontology
            :param prefix: prefix of the ontology
            :param namespace: namespace that prefix is represented
            :return: Ontology
        """
        self.g.parse(location=fpath, format=Ontology.guest_format(fpath))
        if namespace is not None and prefix is not None:
            self.namespaces[prefix] = namespace
            self.reversed_namespaces[namespace] = prefix
            for prefix, namespace in self.namespaces.items():
                self.g.namespace_manager.bind(prefix, namespace, override=True, replace=True)

        return self

    def register_namespace(self, ns: str, uri: str) -> None:
        self.namespaces[ns] = uri
        self.reversed_namespaces[uri] = ns

    def simplify_uri(self, uri: str) -> str:
        if uri.find(':') != -1 and uri.find("://") == -1:
            return uri

        for ns, prefix in self.reversed_namespaces.items():
            if uri.startswith(ns):
                return prefix + uri.replace(ns, ":")

        return self.g.qname(uri)

    def full_uri(self, short_uri: str) -> str:
        if short_uri.find("://") != -1:
            return short_uri

        prefix, path = short_uri.split(':')
        return self.namespaces[prefix] + path

    def in_defined_namespace(self, uri: str) -> bool:
        for prefix, ns in self.namespaces.items():
            if uri.startswith(ns):
                return True
        return False

    def match(self, triple):
        """
            :param triple:
            :return:
        """
        triple = [x if x is None else rdflib.URIRef(x) for x in triple]
        return g.triples(triple)

    def get_classes(self) -> List[str]:
        classes = set()
        is_classes = {OWL.Class, RDFS.Class}
        for s, p, o in self.g:
            if p == RDF.type and o in is_classes:
                classes.add(self.simplify_uri(s))

        return list(classes)

    def get_predicates(self) -> List[str]:
        predicates = set()
        is_predicates = {RDF.Property, OWL.TransitiveProperty, OWL.AnnotationProperty, OWL.DatatypeProperty, OWL.ObjectProperty}
        for s, p, o in self.g:
            if p == RDF.type and o in is_predicates:
                predicates.add(self.simplify_uri(s))

        predicates.add(self.simplify_uri(RDF.value))
        predicates.add(self.simplify_uri(RDFS.label))
        return list(predicates)

    def size(self) -> int:
        """Get number of triples in the ontology"""
        return len(self.g)

    @staticmethod
    def guest_format(fpath: str) -> str:
        """Guest the format of the ontology"""
        _, ext = os.path.splitext(fpath)

        if ext == '.ttl':
            return 'n3'
        if ext == '.rdf' or ext == '.owl':
            return 'xml'
        return ext


class UselessOntology(Ontology):

    def __init__(self) -> None:
        super().__init__("")

    def simplify_uri(self, uri: str) -> str:
        return uri

    def full_uri(self, short_uri: str) -> str:
        return short_uri
