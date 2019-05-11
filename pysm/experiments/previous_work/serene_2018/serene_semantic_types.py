#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

from pathlib import Path

from semantic_modeling.data_io import get_ontology, get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel, SemanticType
from semantic_modeling.utilities.serializable import deserializeCSV


class SereneSemanticTypes:

    def __init__(self, dataset: str, dir: Path) -> None:
        self.ont = get_ontology(dataset)
        self.sms = get_semantic_models(dataset)
        self.sm_prefix_index = {sm.id[:3]: sm for sm in self.sms}
        self.sm_attr2stypes: Dict[str, Dict[str, List[SemanticType]]] = {}
        assert len(self.sm_prefix_index) == len(self.sms), "No duplicated prefix"

        class_uris = set()
        predicates = set()
        for sm in self.sms:
            for n in sm.graph.iter_data_nodes():
                e = n.get_first_incoming_link()
                class_uri = e.get_source_node().label.decode()
                predicate = e.label.decode()

                class_uris.add(class_uri)
                predicates.add(predicate)

        for file in dir.iterdir():
            if file.name.endswith(".df.csv"):
                prefix = file.name[:3]
                self.sm_attr2stypes[prefix] = self.read_serene_stypes(file)
                for attr_lbl, stypes in self.sm_attr2stypes[prefix].items():
                    for stype in stypes:
                        stype.domain = self.recover_class_uris(stype.domain, class_uris)
                        stype.type = self.recover_predicates(stype.type, predicates)

    @staticmethod
    def recover_class_uris(class_uri: str, class_uris: Set[str]) -> str:
        matched = []
        for c in class_uris:
            if c.endswith(class_uri):
                matched.append(c)
        assert len(matched) == 1
        return matched[0]

    @staticmethod
    def recover_predicates(predicate: str, predicates: Set[str]) -> str:
        matched = []
        for c in predicates:
            if c.endswith(predicate):
                matched.append(c)
        assert len(matched) == 1
        return matched[0]

    def semantic_labeling(self,
                          train_sources: List[SemanticModel],
                          test_sources: List[SemanticModel],
                          top_n: int,
                          eval_train: bool = False) -> None:
        """Generate semantic labels and store it in test sources"""
        evaluate_sms = []
        if eval_train:
            evaluate_sms += train_sources
        evaluate_sms += test_sources

        for sm in evaluate_sms:
            prefix = sm.id[:3]
            attr2stypes = self.sm_attr2stypes[prefix]
            for attr in sm.attrs:
                attr.semantic_types = attr2stypes[attr.label][:top_n]

    @staticmethod
    def read_serene_stypes(file: Union[Path, str]) -> Dict[str, List[SemanticType]]:
        content = deserializeCSV(file)
        header = content[0]
        objs = []
        for r in content[1:]:
            objs.append(dict(zip(header, r)))
            assert len(objs[-1]) == len(header), "No duplicated field"

        attr2stypes: Dict[str, List[SemanticType]] = {}
        stypes = [(stype, stype.replace("scores_", "").split("---")) for stype in header[8:] if stype != "scores_unknown"]
        assert header[7] == "user_label"
        for obj in objs:
            attr2stypes[obj['column_name']] = [
                SemanticType(class_uri, predicate, float(obj[stype]))
                for stype, (class_uri, predicate) in stypes
            ]
            attr2stypes[obj['column_name']].sort(key=lambda stype: stype.confidence_score, reverse=True)

        return attr2stypes



if __name__ == '__main__':
    dataset = "museum_edm"
    styper = SereneSemanticTypes(dataset, Path("/workspace/tmp/serene-python-client/datasets/museum_edm/kfold-s01-s14/"))
    styper.read_serene_stypes("/workspace/tmp/serene-python-client/datasets/museum_edm/kfold-s01-s14/s15-s-detroit-institute-of-art.df.csv")
