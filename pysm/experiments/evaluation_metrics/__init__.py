#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import Enum
from typing import NamedTuple, Callable

from data_structure import Graph
from experiments.evaluation_metrics.semantic_labeling import mrr, accuracy, coverage, accuracy_version2
from experiments.evaluation_metrics.semantic_modeling.pyeval import f1_precision_recall, DataNodeMode
from semantic_modeling.karma.semantic_model import SemanticModel


SemanticLabelingEval = NamedTuple("SemanticLabelingEval", [
    ("mrr", Callable[[SemanticModel, SemanticModel], float]),
    ("accuracy", Callable[[SemanticModel, SemanticModel], float]),
    ("coverage", Callable[[SemanticModel, SemanticModel], float]),
])

SemanticModelingEval = NamedTuple("SemanticModelingEval", [
    ("f1_precision_recall", Callable[[Graph, Graph, DataNodeMode], dict]),
    ("stype_acc", Callable[[Graph, Graph], float])
])

stype_eval = SemanticLabelingEval(mrr, accuracy, coverage)
smodel_eval = SemanticModelingEval(f1_precision_recall, accuracy_version2)


class Scenario(Enum):

    SCENARIO_1 = "scenario1"
    SCENARIO_2 = "scenario2"