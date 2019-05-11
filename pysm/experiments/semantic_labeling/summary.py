#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import shutil
import ujson
from pathlib import Path

import numpy as np

# import gmtk.config
# gmtk.config.USE_C_EXTENSION = False

from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.learning.evaluate import predict_sm, evaluate
from datetime import datetime
from experiments.slack_notifier import send_message, ExpResultMessage, TextMessage
from semantic_modeling.assembling.learning.online_learning import create_default_model, online_learning
from semantic_modeling.assembling.learning.shared_models import TrainingArgs
from semantic_modeling.assembling.undirected_graphical_model.model import Model
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_short_train_name
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.serializable import serializeCSV, deserializeCSV


def get_shell_args():
    def str2bool(v):
        assert v.lower() in {"true", "false"}
        return v.lower() == "true"

    parser = argparse.ArgumentParser('Semantic labeling experiment')
    parser.register("type", "boolean", str2bool)

    parser.add_argument('--dataset', type=str, default=None, help="Dataset name")
    parser.add_argument('--exp_dir', type=str, help='Experiment directory, must be existed before')
    parser.add_argument('--channel', type=str)
    parser.add_argument('--detail', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_shell_args()

    dataset = args.dataset
    kfold_eval = [["source", "mrr", "accuracy", "coverage"]]

    exp_dir = Path(args.exp_dir)
    assert exp_dir.exists()

    for file in exp_dir.iterdir():
        if file.name.endswith('.test.csv'):
            eval = deserializeCSV(file)
            assert eval[0][0] == 'source' and eval[-1][0] == 'average'
            kfold_eval += eval[1:-1]

    if len(kfold_eval) == 1:
        print(">>> ERROR NO OUTPUTS")
        send_message(config.slack.channel[args.channel], TextMessage(f"Experiment error: no outputs\n.*Experiment dir*: {args.exp_dir}"))
    else:
        average = [
            'average',
            np.average([float(x[1]) for x in kfold_eval[1:]]),
            np.average([float(x[2]) for x in kfold_eval[1:]]),
            np.average([float(x[3]) for x in kfold_eval[1:]]),
        ]
        kfold_eval.append(average)
        serializeCSV(kfold_eval, exp_dir / f"all.csv")
        send_message(config.slack.channel[args.channel], ExpResultMessage(dataset, args.detail, args.exp_dir, {
            "mrr": average[1],
            "accuracy": average[2],
            "coverage": average[3]
        }))
        print(">>> AVERAGE:", average)