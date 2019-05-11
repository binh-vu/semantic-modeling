#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import chain
from typing import Dict, Tuple, List

from pathlib import Path

from data_structure import Graph, GraphNode
from semantic_labeling.column import Column
from semantic_labeling.column_based_table import ColumnBasedTable
from semantic_labeling.typer import SemanticTyper
from semantic_modeling.settings import Settings
from semantic_modeling.assembling.ont_graph import get_ont_graph
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.assembling.triple_adviser import TripleAdviser, EmpiricalTripleAdviser
from semantic_modeling.config import get_logger, config
from semantic_modeling.data_io import get_ontology, get_semantic_models, get_cache_dir
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import deserialize, serialize


class SemanticTypeAssistant(object):
    """We use semantic type to help justify if class C (not data node) should link to class A or class B
    Score is a potential gain if switching to another class (for example: potential gain if C link to B instead of A (currently C link to A))
    """

    logger = get_logger("app.weak_models.stype_assistant")

    def __init__(self, train_sms: List[SemanticModel], typer: SemanticTyper, triple_adviser: TripleAdviser):
        self.train_sms = {sm.id: sm for sm in train_sms}
        self.stype_db = typer.stype_db
        self.triple_adviser = triple_adviser

        # # contain a mapping from (s, p, o) => table.id, and node which are mounted in SM by o
        # self.parent_stype_index: Dict[Tuple[bytes, bytes, bytes], List[Tuple[str, int]]] = {}
        # for train_sm in train_sms:
        #     for n in train_sm.graph.iter_nodes():
        #         for e in n.iter_outgoing_links():
        #             target = e.get_target_node()
        #             index_key = (n.label, e.label, target.label)
        #             if index_key not in self.parent_stype_index:
        #                 self.parent_stype_index[index_key] = []
        #             self.parent_stype_index[index_key].append((train_sm.id, target.id))

        # contain a mapping from (semantic types & parent stypes (s, p, o) to columns
        self.column_stype_index: Dict[bytes, Dict[Tuple[bytes, bytes, bytes], List[Column]]] = {}
        for train_sm in train_sms:
            table = self.stype_db.get_table_by_name(train_sm.id)

            for dnode in train_sm.graph.iter_data_nodes():
                dlink = dnode.get_first_incoming_link()
                pnode = dlink.get_source_node()
                # stype = (pnode.label, dlink.label)
                plink = pnode.get_first_incoming_link()
                if plink is None:
                    # this is a root node
                    continue

                parent_stype = (plink.get_source_node().label, plink.label, pnode.label)

                if pnode.label not in self.column_stype_index:
                    self.column_stype_index[pnode.label] = {}
                if parent_stype not in self.column_stype_index[pnode.label]:
                    self.column_stype_index[pnode.label][parent_stype] = []

                column = table.get_column_by_name(dnode.label.decode("utf-8"))
                self.column_stype_index[pnode.label][parent_stype].append(column)

        # possible_mount of a node
        self.possible_mounts: Dict[bytes, List[Tuple[bytes, bytes]]] = {}
        for train_sm in train_sms:
            for n in train_sm.graph.iter_class_nodes():
                if n.label not in self.possible_mounts:
                    self.possible_mounts[n.label] = self.triple_adviser.get_subj_preds(n.label)

        # contains the likelihood between 2 columns
        X = self.stype_db.similarity_matrix.reshape((-1, self.stype_db.similarity_matrix.shape[-1]))
        similarity_matrix = typer.model.predict_proba(X)[:, 1]
        self.similarity_matrix = similarity_matrix.reshape(self.stype_db.similarity_matrix.shape[:-1])

        # mapping from column's name to column's index
        self.name2cols: Dict[bytes, Dict[bytes, int]] = {}
        tbl: ColumnBasedTable
        for tbl in chain(self.stype_db.train_tables, self.stype_db.test_tables):
            self.name2cols[tbl.id] = {}
            for col in tbl.columns:
                self.name2cols[tbl.id][col.name.encode('utf-8')] = self.stype_db.col2idx[col.id]

        self.logger.debug("Finish building index for semantic type assistant...")

    def compute_prob(self, sm_id: str, g: Graph) -> Dict[int, float]:
        link2features = {}
        graph_observed_mounts = set()
        graph_observed_class_lbls = set()
        name2col_idx = self.name2cols[sm_id]

        parent_nodes: Dict[int, Tuple[GraphNode, Tuple[bytes, bytes]]] = {}
        for dnode in g.iter_data_nodes():
            dlink = dnode.get_first_incoming_link()
            col_idx = name2col_idx[dnode.label]

            if dlink.source_id not in parent_nodes:
                pnode = dlink.get_source_node()
                plink = pnode.get_first_incoming_link()
                if plink is None:
                    continue

                pstype = (plink.get_source_node().label, plink.label)

                # add pstype to observed mounts
                graph_observed_mounts.add(pstype)
                parent_nodes[dlink.source_id] = (pnode, plink, pstype, [dlink], [col_idx])
            else:
                parent_nodes[dlink.source_id][-2].append(dlink)
                parent_nodes[dlink.source_id][-1].append(col_idx)

        for pnode in g.iter_class_nodes():
            graph_observed_class_lbls.add(pnode.label)

        for pnode, plink, pstype, dlinks, col_idxs in parent_nodes.values():
            # map from possible mount => scores of each columns
            parent_stype_score: Dict[Tuple[bytes, bytes], List[float]] = {}

            # filter out all possible mounts that present in the graph (except the current one),
            # but the domain of the mounts are not in the graph
            possible_mounts = [
                possible_mount for possible_mount in self.possible_mounts.get(pnode.label, [])
                if not ((possible_mount in graph_observed_mounts and possible_mount != pstype)
                        or possible_mount[0] not in graph_observed_class_lbls)
            ]

            if len(possible_mounts) > 1:
                # the number only make sense if there are another place to mount this object to
                for possible_mount in possible_mounts:
                    spo = (possible_mount[0], possible_mount[1], pnode.label)
                    scores = []
                    for i, col_idx in enumerate(col_idxs):
                        # stype = (pnode.label, dlinks[i].label)
                        refcols = self.column_stype_index[pnode.label][spo]
                        best_score = max(
                            self.similarity_matrix[col_idx, self.stype_db.col2idx[refcol.id]] for refcol in refcols)
                        scores.append(best_score)
                    parent_stype_score[possible_mount] = scores

                aggregation_score = {mount: sum(scores) / len(scores) for mount, scores in parent_stype_score.items()}
            else:
                aggregation_score = {}

            if pstype not in aggregation_score:
                link2features[plink.id] = None
            else:
                link2features[plink.id] = aggregation_score.pop(pstype) - max(aggregation_score.values())

        return link2features


_instance = None


def get_stype_assistant_model(dataset: str, train_sms: List[SemanticModel]):
    global _instance
    if _instance is None:
        cache_file = get_cache_dir(dataset, train_sms) / "weak_models" / "stype_assistant.pkl"
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        need_rebuilt = True

        if cache_file.exists():
            SemanticTypeAssistant.logger.debug("Try to load previous run...")
            model, cache_dataset, cache_train_sm_ids = deserialize(cache_file)
            if cache_dataset == dataset and cache_train_sm_ids == {sm.id for sm in train_sms}:
                need_rebuilt = False

            ont_graph = get_ont_graph(dataset)
            ont = get_ontology(dataset)
            stat = Statistic.get_instance(train_sms)
            ota = EmpiricalTripleAdviser(ont_graph, ont, stat.p_triple, 15)
            model.triple_adviser = ota

        if need_rebuilt:
            ont_graph = get_ont_graph(dataset)
            ont = get_ontology(dataset)
            stat = Statistic.get_instance(train_sms)
            ota = EmpiricalTripleAdviser(ont_graph, ont, stat.p_triple, 15)

            typer = SemanticTyper.get_instance(dataset, train_sms)
            try:
                typer.load_model()
            except:
                sms = get_semantic_models(dataset)
                train_ids = {sm.id for sm in train_sms}
                typer.semantic_labeling(train_sms, [sm for sm in sms if sm.id not in train_ids], 4)

            model = SemanticTypeAssistant(train_sms, typer, ota)
            model.triple_adviser = None
            serialize((model, dataset, {sm.id for sm in train_sms}), cache_file)
            model.triple_adviser = ota

        _instance = model

    return _instance
