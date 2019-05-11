#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil
import ujson
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

from experiments.previous_work.mohsen_jws2015.app import MohsenSemanticModeling
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_short_train_name
from semantic_modeling.karma.semantic_model import SemanticModel, SemanticType
from semantic_modeling.utilities.serializable import serializeJSON, deserializeJSON


class MohsenSemanticTyper(object):
    instance = None

    def __init__(self, dataset: str, train_sms: List[SemanticModel]) -> None:
        input_file = Path(config.datasets[dataset].karma_version.as_path()) / "semantic-types" / f"{get_short_train_name(train_sms)}.json"
        if not input_file.exists():
            compute_mohsen_stypes(dataset, train_sms)

        self.stypes = deserializeJSON(input_file)
        self.train_source_ids = {sm.id for sm in train_sms}

    @staticmethod
    def get_instance(dataset: str,
                     train_sms: List[SemanticModel]) -> 'MohsenSemanticTyper':
        if MohsenSemanticTyper.instance is None:
            MohsenSemanticTyper.instance = MohsenSemanticTyper(dataset, train_sms)

        return MohsenSemanticTyper.instance

    def semantic_labeling(self, train_sms: List[SemanticModel],
                          test_sms: List[SemanticModel],
                          top_n: int,
                          eval_train: bool = False) -> None:
        assert set([sm.id for sm in train_sms]) == self.train_source_ids

        if eval_train:
            evaluate_sms = chain(train_sms, test_sms)
        else:
            evaluate_sms = test_sms

        for sm in evaluate_sms:
            pred_stypes = self.stypes[sm.id]
            for attr in sm.attrs:
                if attr.label not in pred_stypes:
                    attr.semantic_types = []
                else:
                    attr.semantic_types = [
                        SemanticType(stype['domain'], stype['type'], stype['confidence_score'])
                        for stype in pred_stypes[attr.label]
                    ][:top_n]


def worker_get_stype(dataset: str, train_sms, test_sm, exec_dir: Path):
    app = MohsenSemanticModeling(dataset, False, True, [sm.id for sm in train_sms], exec_dir=exec_dir)
    return app.semantic_labeling(train_sms, [test_sm])[0]


def compute_mohsen_stypes(dataset: str, train_sms: List[SemanticModel]):
    sms = get_semantic_models(dataset)
    train_sm_ids = [sm.id for sm in train_sms]

    exec_dir = Path(config.fsys.debug.as_path()) / "tmp" / f"mohsen-styper-{get_short_train_name(train_sms)}"
    if exec_dir.exists():
        shutil.rmtree(exec_dir)
    exec_dir.mkdir(exist_ok=True, parents=True)

    semantic_types = {}

    # now we parallel to save time
    # with ThreadPool(os.cpu_count() // 2) as pool:
    with ThreadPool(6) as pool:
        results = {}
        # because karma re-learn semantic types for every data source, we parallel for every data source
        for sm in sms:
            if sm.id in train_sm_ids:
                local_train_sms = [s for s in train_sms if s.id != sm.id]
            else:
                local_train_sms = train_sms

            local_exec_dir = exec_dir / sm.id
            local_exec_dir.mkdir(exist_ok=True)

            results[sm.id] = pool.apply_async(worker_get_stype, (dataset, local_train_sms, sm, local_exec_dir))

        for sid, result in results.items():
            semantic_types[sid] = result.get()

    output_dir = Path(config.datasets[dataset].karma_version.as_path()) / "semantic-types"
    output_dir.mkdir(exist_ok=True)
    serializeJSON(semantic_types, output_dir / f"{get_short_train_name(train_sms)}.json", indent=4)
    return semantic_types


if __name__ == '__main__':
    dataset = "museum_edm"
    train_sms = get_semantic_models(dataset)[:3]
    print(ujson.dumps(compute_mohsen_stypes(dataset, train_sms), indent=4))