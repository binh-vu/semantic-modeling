#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from collections import defaultdict
from typing import Dict, Tuple, List, Set, Union, Optional

from pathlib import Path

from data_structure import GraphNode
from semantic_modeling.config import config
from semantic_modeling.data_io import get_data_tables, get_semantic_models, get_cache_dir
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import serializeJSON, deserializeJSON
from transformation.jsonld_generator import jsonld_generator
"""Predict which field can be use as a pesudo-primary key in case we don't have it"""


class PrimaryKey:

    instance = None

    def __init__(self, pesudo_primary_keys: Dict[bytes, bytes]):
        self.pesudo_primary_keys = pesudo_primary_keys

    def __contains__(self, item):
        return item in self.pesudo_primary_keys

    def __getitem__(self, item):
        return self.pesudo_primary_keys[item]

    @staticmethod
    def get_instance(dataset: str, train_sms: List[SemanticModel]):
        if PrimaryKey.instance is None:
            cache_file = get_cache_dir(dataset, train_sms) / "weak_models" / "primary_keys.json"
            if not cache_file.exists():
                train_sm_ids = {sm.id for sm in train_sms}
                train_tbls = {tbl.id: tbl for tbl in get_data_tables(dataset) if tbl.id in train_sm_ids}
                predictions: Dict[str, List[dict]] = defaultdict(lambda: [])
                pesudo_primary_keys = {}

                for sm in train_sms:
                    jsonld_objects = jsonld_generator(sm, train_tbls[sm.id])
                    for n in sm.graph.iter_class_nodes():
                        fields = [
                            e.label.decode("utf-8")
                            for e in n.iter_outgoing_links()
                            if e.get_target_node().is_data_node()
                        ]
                        if len(fields) == 0:
                            continue
                        if 'karma:classLink' in fields:
                            pesudo_primary_keys[n.label] = 'karma:classLink'
                            continue

                        results = extract_node_data(n, jsonld_objects)
                        views = create_unique_views(results, fields)
                        predictions[n.label].append(predict_pesudo_keys(fields, views))

                for class_lbl, preds in predictions.items():
                    total = defaultdict(lambda: 0)
                    for pred in preds:
                        for link_lbl in pred:
                            total[link_lbl] += pred[link_lbl]
                    for link_lbl, count in total.items():
                        total[link_lbl] = count
                    pesudo_primary_keys[class_lbl] = max(total.items(), key=lambda x: x[1])[0]

                PrimaryKey.instance = PrimaryKey({k: v.encode('utf-8') for k, v in pesudo_primary_keys.items()})
                cache_file.parent.mkdir(exist_ok=True, parents=True)
                serializeJSON(PrimaryKey.instance, cache_file, indent=4)
            else:
                PrimaryKey.instance: PrimaryKey = deserializeJSON(cache_file, Class=PrimaryKey)

        return PrimaryKey.instance

    def to_dict(self):
        return {"pesudo_primary_keys": self.pesudo_primary_keys}

    @staticmethod
    def from_dict(obj: dict):
        return PrimaryKey({k.encode('utf-8'): v.encode('utf-8') for k, v in obj['pesudo_primary_keys'].items()})


def create_unique_views(rows: list, fields: List[str]):
    """Create views for each class objects, default id should be a whole row"""
    views = {}
    for r in rows:
        values = [r[cname] for cname in fields]
        if any(isinstance(x, list) for x in values):
            if all(isinstance(x, list) for x in values) and len({len(x) for x in values}) == 1:
                # all its value is in a list
                for j in range(len(values[0])):
                    key = ",".join(str(values[i][j]) for i in range(len(values)))
                    views[key] = [values[i][j] for i in range(len(values))]
            else:
                # assert False
                key = ",".join((str(x) for x in values))
                views[key] = values
        else:
            key = ",".join((str(x) for x in values))
            views[key] = values
    views = [{cname: r[i] for i, cname in enumerate(fields)} for r in views.values()]
    return views


def extract_node_data(node: GraphNode, rows):

    def get_node_path(node: GraphNode, path):
        link = node.get_first_incoming_link()
        if link is None:
            return path

        return get_node_path(link.get_source_node(), [link.label.decode("utf-8")] + path)

    def extract_data(path, row):
        if len(path) == 0:
            if row is None:
                return []

            object = {}
            for k, v in row.items():
                if k == '@type':
                    continue
                if isinstance(v, dict) or (isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and
                                           '@type' in v[0]):
                    continue
                object[k] = v

            return object

        if isinstance(row[path[0]], list):
            results = []
            for v in row[path[0]]:
                if isinstance(v, list):
                    for v0 in v:
                        res = extract_data(path[1:], v0)
                        if isinstance(res, list):
                            results += res
                        else:
                            results.append(res)
                else:
                    res = extract_data(path[1:], v)
                    if isinstance(res, list):
                        results += res
                    else:
                        results.append(res)
            return results

        return extract_data(path[1:], row[path[0]])

    node_path = get_node_path(node, [])
    results = []
    for row in rows:
        res = extract_data(node_path, row)
        if isinstance(res, list):
            results += res
        else:
            results.append(res)
    return results


def predict_pesudo_keys(fields, views):
    """Simplest prediction, choose the one that have most #n_uniques"""
    uniques = []
    predictions = {}

    for cname in fields:
        n_unique = len({r[cname] for r in views if type(r[cname]) is not list})
        uniques.append(n_unique / len(views))

    maximum = max(uniques)
    for i, field in enumerate(fields):
        if uniques[i] > 0.999 or uniques[i] == maximum:
            predictions[field] = 1
        else:
            predictions[field] = 0

    return predictions
