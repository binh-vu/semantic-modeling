#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from pyutils.progress_utils import Timer

from experiments.evaluation_metrics import DataNodeMode
from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.autolabel.align_graph import align_graph
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.link_prediction.int_graph import IntGraph, add_known_models, IntGraphNode, create_psl_int_graph


def oracle_link_prediction(g: IntGraph, sm: SemanticModel):
    # step 1: find a mapping between g and sm that yield maximum F1 score
    alignment = align_graph(sm.graph, g, DataNodeMode.IGNORE_DATA_NODE)
    bijection = alignment['_bijections'][0]
    true_links = set()

    for attr in sm.attrs:
        dnode = sm.graph.get_node_by_id(attr.id)
        dlink = dnode.get_first_incoming_link()

        if dlink.source_id in bijection.x2prime:
            n: IntGraphNode = g.get_node_by_id(bijection.x2prime[dlink.source_id])
            if any(st.domain.encode("utf-8") == n.label and st.type.encode('utf-8') == dlink.label for st in
                   attr.semantic_types):
                true_links.add((n.readable_id, dlink.label, attr.label))

    for n in sm.graph.iter_nodes():
        for e in n.iter_outgoing_links():
            if e.get_target_node().is_data_node():
                continue

            # if e.source_id in bijection.prime2x and e.target_id in bijection.prime2x:
            if bijection.x2prime.get(e.source_id, None) is not None and bijection.x2prime.get(e.target_id,
                                                                                              None) is not None:
                source = g.get_node_by_id(bijection.x2prime[e.source_id])
                ne = next((ne for ne in source.iter_outgoing_links() if ne.target_id == bijection.x2prime[e.target_id]),
                          None)
                if ne is not None and e.label == ne.label:
                    true_links.add((source.readable_id, e.label, ne.get_target_node().readable_id))

    return true_links


if __name__ == '__main__':
    timer = Timer().start()
    dataset = "museum_edm"
    train_size = 14
    semantic_models = get_semantic_models(dataset)

    semantic_typer = create_semantic_typer(dataset, semantic_models[:train_size])
    semantic_typer.semantic_labeling(semantic_models[:train_size], semantic_models[train_size:], 4, True)

    int_graph = create_psl_int_graph(semantic_models[:train_size])

    print("Preprocessing takes...", timer.lap().report(full_report=True))
    results = oracle_link_prediction(int_graph, semantic_models[3])
    print(ujson.dumps(results, indent=4))
    print("Finish", timer.lap().report(full_report=True))
