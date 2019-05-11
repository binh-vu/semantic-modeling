#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List

from experiments.previous_work import MinhptxSemanticLabeling
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import serializeCSV
from experiments.evaluation_metrics import stype_eval


def eval_sources(sources: List[SemanticModel], foutput: Path):
    eval_hist = [["source", "mrr", "accuracy", "coverage"]]
    for source in sources:
        eval_hist.append([
            source.id,
            stype_eval.mrr(source, source),
            stype_eval.accuracy(source, source),
            stype_eval.coverage(source, source)
        ])

    serializeCSV(eval_hist, foutput)


if __name__ == '__main__':
    train_size = 14
    dataset = "museum_edm"
    source_models: List[SemanticModel] = get_semantic_models(dataset)

    typer = MinhptxSemanticLabeling(dataset)
    typer.semantic_labeling(source_models[:train_size], source_models[train_size:], 4)

    eval_sources(source_models[:train_size], typer.exec_dir / "eval_train.csv")
    eval_sources(source_models[train_size:], typer.exec_dir / "eval_test.csv")

