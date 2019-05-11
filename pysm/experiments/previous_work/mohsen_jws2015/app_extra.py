#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson, logging
from typing import Dict, Tuple, List, Set, Union, Optional

from collections import defaultdict

from pyutils.list_utils import _

from data_structure import Graph, GraphNode, GraphLink, GraphNodeType
from experiments.evaluation_metrics import f1_precision_recall, DataNodeMode
from experiments.evaluation_metrics.semantic_modeling.pyeval import PermutationExploding
from semantic_modeling.karma.karma import KarmaModel
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeJSON, serializeJSON


class SemanticModelAlignment(object):

    def __init__(self, sm: SemanticModel, karma_sm: KarmaModel):
        """Our semantic model and karma semantic model represent same semantic structure, but only name of data nodes are
        changed. We want to build a mapping from data node id in karma_sm to data node id in our sm, such that if we replace
        names of data nodes in karma_sm by name of data nodes in our sm; the new tree still capture same semantic like in
        our sm.

        Our conjecture: semantic of a data node doesn't change when a path from root node to this data node doesn't change.
        """
        try:
            result = f1_precision_recall(sm.graph, karma_sm.graph, DataNodeMode.IGNORE_LABEL_DATA_NODE)
        except PermutationExploding as e:
            logging.error("PermutationExploding at source: %s", sm.id)
            raise

        bijection = result['_bijection']
        try:
            assert result['f1'] == 1.0 and None not in bijection.prime2x and None not in bijection.x2prime and len(bijection.x2prime) == len(bijection.prime2x)
        
            # mapping from karma node's id to our sm node's id
            alignment: Dict[int, int] = {}

            for n_prime in karma_sm.graph.iter_class_nodes():
                n = sm.graph.get_node_by_id(bijection.prime2x[n_prime.id])
                edges: Dict[bytes, List[GraphLink]] = _(n.iter_outgoing_links()).imap(lambda e: (e.label, e)).group_by_key().todict()

                e_primes: List[GraphLink]
                for lbl, e_primes in _(n_prime.iter_outgoing_links()).imap(lambda e: (e.label, e)).group_by_key().get_value():
                    assert len(e_primes) == len(edges[lbl])
                    e_primes = [e for e in e_primes if e.get_target_node().is_data_node()]
                    es = [e for e in edges[lbl] if e.get_target_node().is_data_node()]

                    assert len(e_primes) == len(es)
                    # order doesn't matter because it doesn't change semantic
                    for ep, e in zip(e_primes, es):
                        alignment[ep.target_id] = e.target_id

            self.alignment = alignment
            self.sm = sm
            self.karma_sm = karma_sm
        except Exception as e:
            sm.graph.render2pdf("/tmp/sm.pdf")
            karma_sm.graph.render2pdf("/tmp/karma_sm.pdf")

            logging.error(f"Error when trying to build alignment between models. Source = {sm.id}")
            raise

    def mask_dnode(self, g: Graph) -> Graph:
        """deprecated"""
        g2 = Graph(True, True, True, g.get_n_nodes(), g.get_n_links())
        for n in g.iter_nodes():
            assert g2.add_new_node(n.type, n.label if n.type == GraphNodeType.CLASS_NODE else b"DataNode").id == n.id
        for e in g.iter_links():
            assert g2.add_new_link(e.type, e.label, e.source_id, e.target_id).id == e.id
        return g2

    def load_and_align(self, ont: Ontology, serialized_karma_sm: str) -> KarmaModel:
        """Now, using pre-built alignment, we will transform data node labels of serialized_karma_sm (prediction).
        Assume that order of source_columns in serialilzed_karma_sm must be match with source_columns in karma_sm
        """

        # ensure the assumption
        json_sm = ujson.loads(serialized_karma_sm)
        assert len(json_sm['sourceColumns']) == len(self.karma_sm.source_columns)
        for json_col, col in zip(json_sm['sourceColumns'], self.karma_sm.original_json['sourceColumns']):
            assert col['columnName'] == json_col['columnName']

        for i, json_col in enumerate(json_sm['sourceColumns']):
            col = self.karma_sm.source_columns[i]
            json_col['columnName'] = self.sm.graph.get_node_by_id(self.alignment[col.id]).label.decode("utf-8")

        model = KarmaModel.load_from_string(ont, ujson.dumps(json_sm))

        # double check the result
        for i, col in enumerate(model.source_columns):
            if col.id == -1:
                # this column is missing
                continue

            assert col.column_name == model.graph.get_node_by_id(col.id).label.decode("utf-8")
            assert model.graph.get_node_by_id(col.id).label == self.sm.graph.get_node_by_id(self.alignment[self.karma_sm.source_columns[i].id]).label

        return model