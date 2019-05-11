#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import chain
from typing import List

from pathlib import Path

from experiments.previous_work.mohsen_jws2015 import MohsenSemanticTyper
from experiments.previous_work.serene_2018.serene_semantic_types import SereneSemanticTypes
from semantic_labeling.typer import SemanticTyper
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel, SemanticType
from semantic_modeling.settings import Settings


class OracleSemanticLabeling(object):

    def semantic_labeling(self, train_sources: List[SemanticModel], test_sources: List[SemanticModel], top_n: int, eval_train: bool = False) -> None:
        if eval_train:
            eval_sources = chain(train_sources, test_sources)
        else:
            eval_sources = test_sources

        for test_source in eval_sources:
            for attr in test_source.attrs:
                node = test_source.graph.get_node_by_id(attr.id)
                link = node.get_first_incoming_link()
                attr.semantic_types = [
                    SemanticType(link.get_source_node().label.decode("utf-8"), link.label.decode("utf-8"), confidence_score=1.0)
                ]


class ConstraintOracleSemanticLabeling:

    def semantic_labeling(self, train_sources: List[SemanticModel], test_sources: List[SemanticModel], top_n: int, eval_train: bool = False) -> None:
        known_stypes = set()
        for sm in train_sources:
            for attr in sm.attrs:
                node = sm.graph.get_node_by_id(attr.id)
                link = node.get_first_incoming_link()
                known_stypes.add((link.get_source_node().label, link.label))

        if eval_train:
            eval_sources = chain(train_sources, test_sources)
        else:
            eval_sources = test_sources

        for test_source in eval_sources:
            for attr in test_source.attrs:
                node = test_source.graph.get_node_by_id(attr.id)
                link = node.get_first_incoming_link()
                if (link.get_source_node().label, link.label) in known_stypes:
                    attr.semantic_types = [
                        SemanticType(link.get_source_node().label.decode("utf-8"), link.label.decode("utf-8"), confidence_score=1.0)
                    ]
                else:
                    attr.semantic_types = []


class SemiOracleSemanticLabeling:

    def __init__(self, styper: SemanticTyper) -> None:
        self.styper = styper

    def semantic_labeling(self, train_sources: List[SemanticModel], test_sources: List[SemanticModel], top_n: int, eval_train: bool = False) -> None:
        self.styper.semantic_labeling(train_sources, test_sources, top_n, eval_train)

        if eval_train:
            eval_sources = chain(train_sources, test_sources)
        else:
            eval_sources = test_sources

        for test_source in eval_sources:
            for attr in test_source.attrs:
                node = test_source.graph.get_node_by_id(attr.id)
                link = node.get_first_incoming_link()

                domain = link.get_source_node().label.decode("utf-8")
                type = link.label.decode("utf-8")

                semantic_types = [
                    stype for stype in attr.semantic_types
                    if stype.domain == domain and stype.type == type
                ]

                if len(semantic_types) > 0:
                    attr.semantic_types = semantic_types


def create_semantic_typer(dataset: str, train_sms: List[SemanticModel]) -> SemanticTyper:
    settings = Settings.get_instance()
    if settings.semantic_labeling_method == Settings.MohsenJWS:
        # noinspection PyTypeChecker
        return MohsenSemanticTyper.get_instance(dataset, train_sms)

    if settings.semantic_labeling_method == Settings.ReImplMinhISWC:
        return SemanticTyper.get_instance(dataset, train_sms)

    if settings.semantic_labeling_method == Settings.MohsenJWS + "-Oracle":
        # noinspection PyTypeChecker
        return SemiOracleSemanticLabeling(MohsenSemanticTyper.get_instance(dataset, train_sms))

    if settings.semantic_labeling_method == Settings.ReImplMinhISWC + "-Oracle":
        # noinspection PyTypeChecker
        return SemiOracleSemanticLabeling(SemanticTyper.get_instance(dataset, train_sms))

    if settings.semantic_labeling_method == Settings.OracleSL:
        # noinspection PyTypeChecker
        return OracleSemanticLabeling()

    if settings.semantic_labeling_method == "OracleSL-Constraint":
        # noinspection PyTypeChecker
        return ConstraintOracleSemanticLabeling()

    if settings.semantic_labeling_method == "SereneSemanticType":
        sms = get_semantic_models(dataset)
        if dataset == "museum_edm" and train_sms == sms[:14]:
            serene_dir = "/workspace/tmp/serene-python-client/datasets/GOLD/museum_edm_stypes/kfold-s01-s14"
        elif dataset == "museum_edm" and train_sms == sms[14:]:
            serene_dir = "/workspace/tmp/serene-python-client/datasets/GOLD/museum_edm_stypes/kfold-s15-s28"
        elif dataset == "museum_edm" and train_sms == sms[7:21]:
            serene_dir = "/workspace/tmp/serene-python-client/datasets/GOLD/museum_edm_stypes/kfold-s08-s21"
        elif dataset == "museum_crm" and train_sms == sms[:14]:
            serene_dir = "/workspace/tmp/serene-python-client/datasets/GOLD/museum_crm_stypes/kfold-s01-s14"
        elif dataset == "museum_crm" and train_sms == sms[14:]:
            serene_dir = "/workspace/tmp/serene-python-client/datasets/GOLD/museum_crm_stypes/kfold-s15-s28"
        elif dataset == "museum_crm" and train_sms == sms[7:21]:
            serene_dir = "/workspace/tmp/serene-python-client/datasets/GOLD/museum_crm_stypes/kfold-s08-s21"
        else:
            raise Exception("Invalid configuration of serene semantic types")

        # noinspection PyTypeChecker
        return SereneSemanticTypes(dataset, Path(serene_dir))

    raise Exception(f"Invalid semantic typer: {settings.semantic_labeling_method}")