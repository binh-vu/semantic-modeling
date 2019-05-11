#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
In this experiment, we generate candidates semantic models using Karma approach (both train & test)
train our MRR in train, and test MRR in our test.
"""
import argparse
import shutil
from itertools import chain
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Set
import numpy as np

from experiments.semantic_modeling.kfold_karma import render_graph, run_evaluation_workflow
from pydebug.colorization import colorize_prediction
from experiments.arg_helper import parse_kfold
from experiments.evaluation_metrics import smodel_eval, DataNodeMode, Scenario
from experiments.semantic_labeling.evaluation import eval_sources
from experiments.previous_work.mohsen_jws2015 import MohsenSemanticModeling, SemanticModelAlignment
from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.autolabel.auto_label import AutoLabel
from semantic_modeling.config import config
from semantic_modeling.data_io import get_karma_models, get_semantic_models, get_ontology, get_cache_dir, \
    get_short_train_name
from semantic_modeling.karma.karma import KarmaModel
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import serializeJSON, serializeCSV


def get_eval_hist(sm_ids: Set[str], test_sms, candidate_smss, data_node_mode):
    eval_hist = [["source", "precision", "recall", "f1", "stype-acc"]]
    for gold_sm, candidate_sms in zip(test_sms, candidate_smss):
        if gold_sm.id not in sm_ids:
            continue

        best_candidate_sm = max(candidate_sms, key=lambda candidate_sm:
        smodel_eval.f1_precision_recall(gold_sm.graph, candidate_sm.graph, data_node_mode)['f1'])
        eval_result = smodel_eval.f1_precision_recall(gold_sm.graph, best_candidate_sm.graph, data_node_mode)
        eval_hist.append([gold_sm.id, eval_result["precision"], eval_result["recall"], eval_result["f1"],
                          smodel_eval.stype_acc(gold_sm.graph, best_candidate_sm.graph)])

    eval_hist.append([
        'average',
        np.average([float(x[1]) for x in eval_hist[1:]]),
        np.average([float(x[2]) for x in eval_hist[1:]]),
        np.average([float(x[3]) for x in eval_hist[1:]]),
        np.average([float(x[4]) for x in eval_hist[1:]])
    ])

    return eval_hist


def create_rust_input(dataset: str, scenario: Scenario, train_sms, test_sms):
    train_sm_ids = [sm.id for sm in train_sms]
    exec_dir = get_cache_dir(dataset, train_sms) / "mohsen_jws2015"
    modeler = MohsenSemanticModeling(
        dataset, False, False, train_sm_ids,
        exec_dir=exec_dir,
        sm_type_dir=Path(config.fsys.debug.as_path()) / "tmp" / "models-json-temp"
    )

    candidate_smss = modeler.sm_candidate_generation(train_sms, test_sms)
    if scenario == Scenario.SCENARIO_1:
        data_node_mode = DataNodeMode.IGNORE_DATA_NODE
    else:
        data_node_mode = DataNodeMode.NO_TOUCH

    train_sm_ids = {sm.id for sm in train_sms}
    real_test_sm_ids = {sm.id for sm in test_sms if sm.id not in train_sm_ids}

    train_eval_hist = get_eval_hist(train_sm_ids, test_sms, candidate_smss, data_node_mode)
    test_eval_hist = get_eval_hist(real_test_sm_ids, test_sms, candidate_smss, data_node_mode)

    serializeCSV(train_eval_hist, exec_dir / f"evaluation_result_{scenario.value}.train.oracle.csv")
    serializeCSV(test_eval_hist, exec_dir / f"evaluation_result_{scenario.value}.test.oracle.csv")

    # now create rust bridge
    obj = {}
    for gold_sm, candidate_sms in zip(test_sms, candidate_smss):
        obj[gold_sm.id] = [c.graph.to_dict() for c in candidate_sms]
    serializeJSON(obj, exec_dir / "rust-karma-pred-input.json")


def get_shell_args():
    parser = argparse.ArgumentParser('Mohsen semantic modeling experiment')

    # copied from settings.py
    parser.add_argument('--dataset', type=str, default="museum_edm", help="Dataset name")
    parser.add_argument('--kfold', type=str, default='{"train_sm_ids": ["s00-s05"], "test_sm_ids": ["s06-s06"]}', help='kfold json object of {train_sm_ids: [], test_sm_ids: []}')
    parser.add_argument('--semantic_labeling_top_n_stypes', type=int, default=1, help='Number of semantic types')
    parser.add_argument('--semantic_typer', type=str, help='kind of semantic type')
    # parser.add_argument('--top_n_semantic_modeling', type=str, help='Top N candidates')
    parser.add_argument('--exp_dir', type=str, help='Experiment directory, must be existed before')

    args = parser.parse_args()
    try:
        assert args.dataset is not None
        args.kfold = parse_kfold(args.dataset, args.kfold)
    except AssertionError:
        parser.print_help()
        raise

    return args


if __name__ == '__main__':
    # HYPER-ARGS
    args = get_shell_args()

    Settings.get_instance(False).semantic_labeling_top_n_stypes = args.semantic_labeling_top_n_stypes
    Settings.get_instance().semantic_labeling_method = args.semantic_typer
    Settings.get_instance().log_current_settings()

    exp_dir = Path(args.exp_dir)
    assert exp_dir.exists()

    source_models = {sm.id: sm for sm in get_semantic_models(args.dataset)}
    train_sms = [source_models[sid] for sid in args.kfold['train_sm_ids']]
    # including train in test because we want to generate training candidate semantic models as well
    test_sms = train_sms + [source_models[sid] for sid in args.kfold['test_sm_ids']]

    eval_hist = run_evaluation_workflow(args.dataset, Scenario.SCENARIO_2, train_sms, test_sms)
    create_rust_input(args.dataset, Scenario.SCENARIO_2, train_sms, test_sms)
