#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Optional, Set

from data_structure import Graph, GraphLink
from experiments.evaluation_metrics import DataNodeMode
from semantic_modeling.assembling.autolabel.align_graph import align_graph
# from semantic_modeling.assembling.undirected_graphical_model.model_core import numbering_link_labels
# from semantic_modeling.assembling.undirected_graphical_model.model_extra import get_numbered_link_label


def get_gold_triples(gold_sm: Graph, is_blurring_label: bool) -> Set[Tuple[int, bytes, Union[bytes, int]]]:
    gold_triples = set()
    for node in gold_sm.iter_class_nodes():
        outgoing_links: List[GraphLink] = list(node.iter_outgoing_links())
        numbered_links = numbering_link_labels(outgoing_links)

        for link in outgoing_links:
            dest_node = link.get_target_node()
            if dest_node.is_class_node():
                dest_label = link.target_id
            else:
                dest_label = get_numbered_link_label(
                    "DATA_NODE", numbered_links[link.id]) if is_blurring_label else dest_node.label

            triple = (link.source_id, link.label, dest_label)
            gold_triples.add(triple)
    return gold_triples


def max_f1(gold_sm: Graph, pred_sm: Graph, is_blurring_label: bool, gold_triples: Set[Tuple[int, bytes, Union[bytes, int]]]):
    alignment = align_graph(gold_sm, pred_sm, DataNodeMode.IGNORE_LABEL_DATA_NODE if is_blurring_label else DataNodeMode.NO_TOUCH)
    bijection = alignment['_bijections'][0]
    link2label = {}

    # build example from this candidate model
    for node in pred_sm.iter_class_nodes():
        outgoing_links = list(node.iter_outgoing_links())
        numbered_links = numbering_link_labels(outgoing_links)

        for link in outgoing_links:
            dest_node = link.get_target_node()
            if dest_node.is_class_node():
                dest_label = bijection.prime2x[link.target_id]
            else:
                dest_label = get_numbered_link_label(
                    "DATA_NODE", numbered_links[link.id]) if is_blurring_label else dest_node.label

            triple = (bijection.prime2x[link.source_id], link.label, dest_label)
            link2label[link.id] = triple in gold_triples

    return link2label, bijection.prime2x, alignment['f1']


def max_f1_no_ambiguous(gold_sm: Graph, pred_sm: Graph, is_blurring_label: bool, gold_triples: Set[Tuple[int, bytes, Union[bytes, int]]]):
    alignment = align_graph(gold_sm, pred_sm, DataNodeMode.IGNORE_LABEL_DATA_NODE if is_blurring_label else DataNodeMode.NO_TOUCH)
    if len(alignment['_bijections']) != 1:
        return None, None, None

    bijection = alignment['_bijections'][0]
    link2label = {}

    # build example from this candidate model
    for node in pred_sm.iter_class_nodes():
        outgoing_links = list(node.iter_outgoing_links())
        numbered_links = numbering_link_labels(outgoing_links)

        for link in outgoing_links:
            dest_node = link.get_target_node()
            if dest_node.is_class_node():
                dest_label = bijection.prime2x[link.target_id]
            else:
                dest_label = get_numbered_link_label(
                    "DATA_NODE", numbered_links[link.id]) if is_blurring_label else dest_node.label

            triple = (bijection.prime2x[link.source_id], link.label, dest_label)
            link2label[link.id] = triple in gold_triples

    return link2label, bijection.prime2x, alignment['f1']

# Copied from model_core and model_extra
def numbering_link_labels(links: List[GraphLink]) -> Dict[int, int]:
    accum_numbered_links = {}
    numbered_links = {}

    for l in links:
        if l.label not in accum_numbered_links:
            accum_numbered_links[l.label] = 1
        else:
            accum_numbered_links[l.label] += 1

    for l in links:
        numbered_links[l.id] = accum_numbered_links[l.label]
        accum_numbered_links[l.label] -= 1

    return numbered_links

def get_numbered_link_label(link_label: str, number: int) -> str:
    """Number a link"""
    return "%s:_%d" % (link_label, number)