#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from typing import *

from pathlib import Path

from experiments.arg_helper import get_sm_ids_by_name_range
from experiments.previous_work.mohsen_jws2015 import MohsenSemanticTyper
from semantic_labeling import create_semantic_typer, SemanticTyper
from semantic_modeling.data_io import get_ontology, get_semantic_models
from semantic_modeling.settings import Settings


def get_serene_style(class_uri, predicate):
    ns, domain = class_uri.split(":")
    ns, type = predicate.split(":")

    return "scores_%s---%s" % (domain, type)

if __name__ == '__main__':
    dataset = "museum_edm"
    sms = get_semantic_models(dataset)
    ont = get_ontology(dataset)

    serene_dir = Path("/workspace/tmp/serene-python-client/datasets/") / dataset
    top_n = 4
    normalize_score = False
    for semantic_type in ["MohsenJWS", "ReImplMinhISWC", "OracleSL-Constraint"]:
        Settings.get_instance(False).semantic_labeling_method = semantic_type

        for kfold in ["kfold-s01-s14", "kfold-s15-s28", "kfold-s08-s21"]:
            serene_stypes = {}
            for sm in sms:
                for n in sm.graph.iter_data_nodes():
                    e = n.get_first_incoming_link()
                    stype = get_serene_style(e.get_source_node().label.decode(), e.label.decode())
                    if stype not in serene_stypes:
                        serene_stypes[stype] = len(serene_stypes)
            serene_stypes["scores_unknown"] = len(serene_stypes)

            header = ["", "column_id", "column_name", "confidence", "dataset_id", "model_id", "label", "user_label"]
            header_offset = len(header)
            for stype, idx in sorted(serene_stypes.items(), key=lambda x: x[1]):
                assert len(header) == idx + header_offset
                header.append(stype)

            train_sm_ids = {sm for sm in get_sm_ids_by_name_range(*kfold.replace("kfold-", "").split("-"), [sm.id for sm in sms])}

            train_sms = [sm for sm in sms if sm.id in train_sm_ids]
            test_sms = [sm for sm in sms if sm.id not in train_sm_ids]

            MohsenSemanticTyper.instance = None
            SemanticTyper.instance = None
            typer = create_semantic_typer(dataset, train_sms)
            print(typer.__class__)
            typer.semantic_labeling(train_sms, test_sms, top_n, False)

            predicted_stypes = {}
            for sm in test_sms:
                rows = [header]
                for i, attr in enumerate(sm.attrs):
                    if normalize_score:
                        norm = sum((x.confidence_score for x in attr.semantic_types))
                    else:
                        norm = 1

                    if len(attr.semantic_types) == 0:
                        continue

                    row = [i, None, attr.label, attr.semantic_types[0].confidence_score / norm, None, None, get_serene_style(attr.semantic_types[0].domain, attr.semantic_types[0].type), None]
                    while len(row) < len(header):
                        row.append(0.0)

                    for stype in attr.semantic_types:
                        idx = serene_stypes[get_serene_style(stype.domain, stype.type)] + header_offset
                        row[idx] = stype.confidence_score / norm
                    rows.append(row)
                predicted_stypes[sm.id[:3]] = rows

            (serene_dir / kfold).mkdir(exist_ok=True, parents=True)
            with open(serene_dir / kfold / ("%s_%s_stypes.json" % (semantic_type, normalize_score)), "w") as f:
                ujson.dump(predicted_stypes, f, indent=4)