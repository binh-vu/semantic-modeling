#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional

from data_structure import Graph
from experiments.evaluation_metrics import DataNodeMode
from semantic_modeling.assembling.autolabel.align_graph import align_graph
from semantic_modeling.assembling.autolabel.maxf1 import numbering_link_labels, get_numbered_link_label


def preserved_structure(gold_sm: Graph, pred_sm: Graph, gold_triples: Set[Tuple[int, bytes, Union[bytes, int]]]
                         ) -> Tuple[Dict[int, bool], Dict[int, Optional[int]]]:
    alignment = align_graph(gold_sm, pred_sm, DataNodeMode.IGNORE_DATA_NODE)
    bijections = alignment['_bijections']
    best_bijection = None
    best_link2label = None
    best_score = -1

    # build example from this candidate model
    for bijection in bijections:
        link2label = {}
        for node in pred_sm.iter_class_nodes():
            outgoing_links = list(node.iter_outgoing_links())
            for link in outgoing_links:
                dest_node = link.get_target_node()
                if dest_node.is_class_node():
                    dest_label = bijection.prime2x[link.target_id]
                else:
                    dest_label = dest_node.label

                triple = (bijection.prime2x[link.source_id], link.label, dest_label)
                link2label[link.id] = triple in gold_triples
        score = sum(link2label.values())
        if score > best_score:
            best_score = score
            best_bijection = bijection
            best_link2label = link2label

    return best_link2label, best_bijection.prime2x
