#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, Tuple, List, Union, Optional

from data_structure import Graph
from semantic_modeling.karma.semantic_model import SemanticModel


def mrr(gold_sm: SemanticModel, pred_sm: SemanticModel) -> float:
    """The learned semantic types should be stored in pred_sm.attributes"""
    ranks = []
    for attr in gold_sm.attrs:
        pred_attr = pred_sm.label2attrs[attr.label]
        if not gold_sm.graph.has_node_with_id(attr.id):
            # this column is ignored, and not used in sm
            if len(pred_attr.semantic_types) == 0:
                ranks.append(1)
            else:
                ranks.append(0)
        else:
            node = gold_sm.graph.get_node_by_id(attr.id)
            assert node.n_incoming_links == 1, "Not support multi-parents"

            link = node.get_first_incoming_link()
            gold_st_domain = link.get_source_node().label.decode("utf-8")
            gold_st_type = link.label.decode("utf-8")

            for i, pred_st in enumerate(pred_attr.semantic_types):
                if pred_st.domain == gold_st_domain and pred_st.type == gold_st_type:
                    ranks.append(1.0 / (i + 1))
                    break
            else:
                ranks.append(0)

    return np.average(ranks)


def accuracy(gold_sm: SemanticModel, pred_sm: SemanticModel) -> float:
    """The learned semantic types should be stored in pred_sm.attributes"""
    ranks = []
    for attr in gold_sm.attrs:
        pred_attr = pred_sm.label2attrs[attr.label]
        if not gold_sm.graph.has_node_with_id(attr.id):
            # this column is ignored, and not used in sm
            if len(pred_attr.semantic_types) == 0:
                ranks.append(1)
            else:
                ranks.append(0)
        else:
            node = gold_sm.graph.get_node_by_id(attr.id)
            assert node.n_incoming_links == 1, "Not support multi-parents"

            link = node.get_first_incoming_link()
            gold_st_domain = link.get_source_node().label.decode("utf-8")
            gold_st_type = link.label.decode("utf-8")

            if len(pred_attr.semantic_types) == 0:
                ranks.append(0)
            else:
                pred_st = pred_attr.semantic_types[0]
                if pred_st.domain == gold_st_domain and pred_st.type == gold_st_type:
                    ranks.append(1)
                else:
                    ranks.append(0)

    return np.average(ranks)


def accuracy_version2(gold_sm: Graph, pred_sm: Graph) -> float:
    dnodes = {}
    for dnode in gold_sm.iter_data_nodes():
        dlink = dnode.get_first_incoming_link()
        dnodes[dnode.label] = (dlink.get_source_node().label, dlink.label)

    assert len(dnodes) == sum(1 for n in gold_sm.iter_data_nodes()), "Label of data nodes must be unique"
    prediction = {}
    for dnode in pred_sm.iter_data_nodes():
        dlink = dnode.get_first_incoming_link()
        if dlink is None:
            continue
            
        if (dlink.get_source_node().label, dlink.label) == dnodes[dnode.label]:
            prediction[dnode.label] = 1
        else:
            prediction[dnode.label] = 0

    for lbl in dnodes:
        if lbl not in prediction:
            prediction[lbl] = 0

    assert len(prediction) == len(dnodes)
    return sum(prediction.values()) / len(prediction)


def coverage(gold_sm: SemanticModel, pred_sm: SemanticModel) -> float:
    """The learned semantic types should be stored in pred_sm.attributes"""
    ranks = []
    for attr in gold_sm.attrs:
        pred_attr = pred_sm.label2attrs[attr.label]
        if not gold_sm.graph.has_node_with_id(attr.id):
            # this column is ignored, and not used in sm
            if len(pred_attr.semantic_types) == 0:
                ranks.append(1)
            else:
                ranks.append(0)
        else:
            node = gold_sm.graph.get_node_by_id(attr.id)
            assert node.n_incoming_links == 1, "Not support multi-parents"

            link = node.get_first_incoming_link()
            gold_st_domain = link.get_source_node().label.decode("utf-8")
            gold_st_type = link.label.decode("utf-8")

            if len(pred_attr.semantic_types) == 0:
                ranks.append(0)
            else:
                for pred_st in pred_attr.semantic_types:
                    if pred_st.domain == gold_st_domain and pred_st.type == gold_st_type:
                        ranks.append(1)
                        break
                else:
                    ranks.append(0)

    return np.average(ranks)
