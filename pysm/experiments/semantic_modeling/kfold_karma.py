#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Evaluate scenario 2, with input of semantic labeling from minhptx_iswc2016"""
import argparse
import shutil
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List
import numpy as np

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


def run_evaluation_workflow(dataset: str, scenario: Scenario, train_sms, test_sms):
    ont: Ontology = get_ontology(dataset)
    karma_models: List[KarmaModel] = get_karma_models(dataset)
    semantic_models: List[SemanticModel] = get_semantic_models(dataset)
    train_sm_ids = [sm.id for sm in train_sms]

    sdesc_args = dict(
        dataset=dataset,
        train_sm_ids=train_sm_ids,
        use_correct_type=False,  # we always put semantic types to learnedSemanticTypes, even for userSetSemanticTypes
        use_old_semantic_typer=False,
        exec_dir=get_cache_dir(dataset, train_sms) / "mohsen_jws2015",
        sm_type_dir=Path(config.fsys.debug.as_path()) / "tmp" / "models-json-temp"
    )
    # STEP 1: run semantic typing to generate semantic typing and put result to a temporal folder
    if sdesc_args['sm_type_dir'].exists():
        shutil.rmtree(sdesc_args['sm_type_dir'])
    sdesc_args['sm_type_dir'].mkdir(exist_ok=True, parents=True)

    top_k_types = Settings.get_instance().semantic_labeling_top_n_stypes
    typer = create_semantic_typer(dataset, train_sms)
    typer.semantic_labeling(train_sms, test_sms, top_k_types, eval_train=True)

    for sm, ksm in zip(semantic_models, karma_models):
        # assign semantic types to learnedSemanticTypes
        sm_alignment = SemanticModelAlignment(sm, ksm)
        for col in ksm.source_columns:
            attr = sm.get_attr_by_label(
                sm.graph.get_node_by_id(sm_alignment.alignment[col.id]).label.decode('utf-8'))
            node = ksm.karma_graph.get_node_by_id(col.id)
            link = node.get_first_incoming_link()

            node.learned_semantic_types = [
                KarmaSemanticType(node.id, stype.domain, stype.type, typer.__class__.__name__, stype.confidence_score)
                for stype in attr.semantic_types
            ]
            node.user_semantic_types = [
                KarmaSemanticType(node.id, link.get_source_node().label.decode(), link.label.decode(), "User", 1.0)
            ]

        serializeJSON(ksm.to_normalized_json_model(ont), sdesc_args['sm_type_dir'] / f"{ksm.id}-model.json", indent=4)

    # STEP 2: invoking semantic modeling
    modeler = MohsenSemanticModeling(**sdesc_args)
    pred_sms = modeler.sm_prediction(train_sms, test_sms)

    # STEP 3: prediction semantic mapping result
    eval_hist = [["source", "precision", "recall", "f1", "stype-acc"]]
    if scenario == Scenario.SCENARIO_1:
        data_node_mode = DataNodeMode.IGNORE_DATA_NODE
    else:
        data_node_mode = DataNodeMode.NO_TOUCH

    for sm, pred_sm in zip(test_sms, pred_sms):
        eval_result = smodel_eval.f1_precision_recall(sm.graph, pred_sm.graph, data_node_mode)
        eval_hist.append([sm.id, eval_result["precision"], eval_result["recall"], eval_result["f1"], smodel_eval.stype_acc(sm.graph, pred_sm.graph)])

    eval_hist.append([
        'average',
        np.average([float(x[1]) for x in eval_hist[1:]]),
        np.average([float(x[2]) for x in eval_hist[1:]]),
        np.average([float(x[3]) for x in eval_hist[1:]]),
        np.average([float(x[4]) for x in eval_hist[1:]])
    ])
    serializeCSV(eval_hist, sdesc_args["exec_dir"] / f"evaluation_result_{scenario.value}.csv")

    # STEP 4: prediction semantic labeling result
    pred_stypes = modeler.semantic_labeling(train_sms, test_sms)
    for pred_stype, sm in zip(pred_stypes, test_sms):
        for attr in sm.attrs:
            if attr.label not in pred_stype:
                attr.semantic_types = []
            else:
                attr.semantic_types = pred_stype[attr.label]
    eval_sources(test_sms, sdesc_args["exec_dir"] / f"evaluation_result_{scenario.value}_stype.csv")

    # STEP 5: visualize the prediction
    (sdesc_args['exec_dir'] / "prediction-viz").mkdir(exist_ok=True)
    need_render_graphs = [(colorize_prediction(pred_sm.graph,
                                               AutoLabel.auto_label_max_f1(sm.graph, pred_sm.graph, False)[0]),
                           sdesc_args['exec_dir'] / "prediction-viz" / f"{sm.id}.png")
                          for sm, pred_sm in zip(test_sms, pred_sms)]
    with ThreadPool(32) as p:
        p.map(render_graph, need_render_graphs)

    return eval_hist


def render_graph(args):
    g, fpath = args
    g.render2img(fpath)


def get_shell_args():
    parser = argparse.ArgumentParser('Mohsen semantic modeling experiment')

    # copied from settings.py
    parser.add_argument('--dataset', type=str, default="museum_edm", help="Dataset name")
    parser.add_argument('--kfold', type=str, default='{"train_sm_ids": ["s00-s05"], "test_sm_ids": ["s06-s06"]}', help='kfold json object of {train_sm_ids: [], test_sm_ids: []}')
    parser.add_argument('--semantic_labeling_top_n_stypes', type=int, default=1, help='Number of semantic types')
    parser.add_argument('--semantic_typer', type=str, help='kind of semantic type')
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
    test_sms = [source_models[sid] for sid in args.kfold['test_sm_ids']]

    eval_hist = run_evaluation_workflow(args.dataset, Scenario.SCENARIO_2, train_sms, test_sms)
    serializeCSV(eval_hist, exp_dir / f"kfold-{get_short_train_name(train_sms)}.test.csv")
    serializeJSON(args, exp_dir / f"kfold-{get_short_train_name(train_sms)}.meta.json", indent=4)
    shutil.move(get_cache_dir(args.dataset, train_sms) / "mohsen_jws2015", exp_dir / f"kfold-{get_short_train_name(train_sms)}")
