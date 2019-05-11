#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *
import matplotlib.pyplot as plt

from experiments.evaluation_metrics import smodel_eval
from experiments.previous_work.mohsen_jws2015 import MohsenSemanticModeling
from experiments.semantic_labeling.evaluation import get_shell_args
from semantic_labeling import create_semantic_typer
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.settings import Settings

if __name__ == "__main__":
    dataset = "museum_edm"
    styper = Settings.ReImplMinhISWC
    exec_dir = "/home/rook/Downloads/tmp/kfold-s01-to-s14"
    sm_type_dir = ""

    source_models: List[SemanticModel] = get_semantic_models(dataset)
    train_sms = source_models[:14]
    test_sms = source_models[14:]

    Settings.get_instance(False).semantic_labeling_method = styper
    Settings.get_instance().log_current_settings()

    typer = create_semantic_typer(dataset, train_sms)
    typer.semantic_labeling(train_sms, test_sms, 4, eval_train=False)

    modeler = MohsenSemanticModeling(dataset, False, False, [sm.id for sm in train_sms], exec_dir, None)
    pred_sms = modeler.sm_prediction(train_sms, test_sms)

    correct_dnodes = []
    incorrect_dnodes = []
    correct_scores = []
    incorrect_scores = []

    for sm, pred_sm in zip(test_sms, pred_sms):
        for dnode in pred_sm.graph.iter_data_nodes():
            attr = sm.get_attr_by_label(dnode.label.decode())
            gold_dnode = sm.graph.get_node_by_id(attr.id)
            gold_link = gold_dnode.get_first_incoming_link()

            link = dnode.get_first_incoming_link()
            if gold_link.label == link.label and gold_link.get_source_node().label == link.get_source_node().label:
                correct_dnodes.append(dnode.label.decode())
                # for stype in attr.semantic_types:
                #     if link.label.decode() == stype.type and link.get_source_node().label.decode() == stype.domain:
                #         correct_scores.append(stype.confidence_score)
                #     else:
                #         incorrect_scores.append(stype.confidence_score)
            else:
                incorrect_dnodes.append(dnode.label.decode())
                for stype in attr.semantic_types:
                    if link.label.decode() == stype.type and link.get_source_node().label.decode() == stype.domain:
                        incorrect_scores.append(stype.confidence_score)
                    elif gold_link.label.decode() == stype.type and gold_link.get_source_node().label.decode() == stype.domain:
                        correct_scores.append(stype.confidence_score)

        eval_result = smodel_eval.f1_precision_recall(sm.graph, pred_sm.graph, 0)
        print([sm.id, eval_result["precision"], eval_result["recall"], eval_result["f1"]])

    print(len(incorrect_dnodes))
    # correct_scores = []
    # incorrect_scores = []
    # for sm in test_sms:
    #     for attr in sm.attrs:
    #         for stype in attr.semantic_types:
    #             link = sm.graph.get_node_by_id(attr.id).get_first_incoming_link()
    #             if link.label.decode() == stype.type and link.get_source_node().label.decode() == stype.domain:
    #                 correct_scores.append(stype.confidence_score)
    #             else:
    #                 incorrect_scores.append(stype.confidence_score)

    print(correct_scores)
    print(incorrect_scores)
    # plt.hist(correct_scores, 20)
    # plt.show()
    #
    # plt.hist(incorrect_scores, 20)
    # plt.show()

    plt.hist([x - y for x, y in zip(correct_scores, incorrect_scores)], 10)
    plt.show()
    print("HA")