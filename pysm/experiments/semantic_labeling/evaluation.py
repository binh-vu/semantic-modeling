#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
from typing import List

from pathlib import Path

import numpy

from experiments.arg_helper import parse_kfold
from experiments.evaluation_metrics import stype_eval
from semantic_labeling import create_semantic_typer
from semantic_modeling.data_io import get_semantic_models, get_short_train_name
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.serializable import serializeCSV


def eval_sources(sources: List[SemanticModel], foutput: Path):
    eval_hist = [["source", "mrr", "accuracy", "coverage"]]
    for source in sources:
        eval_hist.append([
            source.id,
            stype_eval.mrr(source, source),
            stype_eval.accuracy(source, source),
            stype_eval.coverage(source, source)
        ])
    eval_hist.append([
        "average",
        numpy.average([x[1] for x in eval_hist[1:]]),
        numpy.average([x[2] for x in eval_hist[1:]]),
        numpy.average([x[3] for x in eval_hist[1:]])
    ])

    serializeCSV(eval_hist, foutput)
    return eval_hist


def get_shell_args():
    parser = argparse.ArgumentParser('Semantic Labeling experiment')

    # copied from settings.py
    parser.add_argument('--dataset', type=str, default=None, help="Dataset name")
    parser.add_argument('--kfold', type=str, default='{"train_sm_ids": ["s01-s03"], "test_sm_ids": ["s02-s02"]}',
                        help='kfold json object of {train_sm_ids: [], test_sm_ids: []}')
    parser.add_argument('--semantic_typer', type=str,
                        help='Type of semantic typer')
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
    args = get_shell_args()
    source_models: List[SemanticModel] = get_semantic_models(args.dataset)
    train_sms = [sm for sm in source_models if sm.id in args.kfold['train_sm_ids']]
    test_sms = [sm for sm in source_models if sm.id in args.kfold['test_sm_ids']]

    Settings.get_instance(False).semantic_labeling_method = args.semantic_typer
    Settings.get_instance().log_current_settings()

    typer = create_semantic_typer(args.dataset, train_sms)
    typer.semantic_labeling(train_sms, test_sms, 4, eval_train=True)

    exp_dir = Path(args.exp_dir)
    eval_sources(train_sms, exp_dir / f"{typer.__class__.__name__}_{get_short_train_name(train_sms)}_eval.train.csv")
    eval_sources(test_sms, exp_dir / f"{typer.__class__.__name__}_{get_short_train_name(train_sms)}_eval.test.csv")
