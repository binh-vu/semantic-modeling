#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Optional, Union

import ujson

import os

from data_structure import Graph, GraphNodeType
from experiments.previous_work.mohsen_jws2015.app_extra import SemanticModelAlignment
from experiments.previous_work.mohsen_jws2015.setup_karma import setup_karma, execute_karma_code
from semantic_modeling.config import config, get_logger
from semantic_modeling.data_io import get_ontology, get_karma_models, get_cache_dir
from semantic_modeling.karma.karma import KarmaModel
from semantic_modeling.karma.karma_graph import KarmaGraph
from semantic_modeling.karma.karma_node import KarmaSemanticType, KarmaGraphNode
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.settings import Settings


class MohsenSemanticModeling(object):
    logger = get_logger("app.mohsen_jws2015")

    def __init__(self, dataset: str, use_correct_type: bool, use_old_semantic_typer: bool, train_sm_ids: List[str],
                 exec_dir: Optional[Union[str, Path]] = None, sm_type_dir: Optional[Union[str, Path]] = None):
        self.dataset: str = dataset
        self.train_sm_ids = train_sm_ids
        self.ont = get_ontology(dataset)
        self.karma_models: Dict[str, KarmaModel] = {km.id: km for km in get_karma_models(dataset)}

        # can only run once time, trying re-invoke will generate an error
        self.__has_run_modeling = False
        if exec_dir is None:
            exec_dir = get_cache_dir(dataset, train_sm_ids) / "mohsen_jws2015"
        self.exec_dir: Path = Path(exec_dir)
        self.sm_type_dir = sm_type_dir

        # parameters for mohsen's algorithm
        self.use_old_semantic_typer = use_old_semantic_typer
        self.use_correct_type = use_correct_type
        assert Settings.get_instance().semantic_labeling_top_n_stypes <= 4
        self.num_candidate_semantic_type = 4
        self.multiple_same_property_per_node = True

        self.coherence = 1.0
        self.confidence = 1.0
        self.size_reduction = 0.5

        self.num_candidate_mappings = 50
        self.mapping_branching_factor = 50
        self.topk_steiner_tree = 10

        # take all, not cut off everything
        self.cut_off = int(1e6)
        self.our_and_karma_sm_alignments = {}

    def get_meta(self, train_source_names: List[str], test_source_names: List[str]) -> Dict:
        return {
            "dataset": self.dataset,
            "use_correct_type": self.use_correct_type,
            "use_old_semantic_typer": self.use_old_semantic_typer,
            "num_candidate_semantic_type": self.num_candidate_semantic_type,
            "multiple_same_property_per_node": self.multiple_same_property_per_node,
            "coherence": self.coherence,
            "confidence": self.confidence,
            "size_reduction": self.size_reduction,
            "num_candidate_mappings": self.num_candidate_mappings,
            "mapping_branching_factor": self.mapping_branching_factor,
            "topk_steiner_tree": self.topk_steiner_tree,
            "train_source_names": train_source_names,
            "test_source_names": test_source_names,
            "cut_off": self.cut_off
        }

    def init(self, train_source_names: List[str], test_source_names: List[str]):
        if self.__has_run_modeling:
            raise Exception("Cannot call init twice!!")

        train_source_names = sorted(train_source_names)
        test_source_names = sorted(test_source_names)
        assert self.train_sm_ids == train_source_names

        execution_meta_file = self.exec_dir / "execution-meta.json"
        lock_file = self.exec_dir / "lock.pid"

        if lock_file.exists():
            raise Exception("Cannot run mohsen method because another process is running")

        if execution_meta_file.exists():
            # only have this file when previous execution is success!
            self.logger.debug("Load information from previous run...")
            re_executing = False
            with open(execution_meta_file, 'r') as f:
                try:
                    meta = ujson.load(f)
                except ValueError:
                    re_executing = True

                if re_executing is False:
                    test_source_names = set(meta.pop("test_source_names"))
                    new_meta = self.get_meta(train_source_names, test_source_names)
                    if test_source_names.difference(set(new_meta.pop("test_source_names"))):
                        re_executing = True
                    else:
                        re_executing = meta != new_meta
        else:
            re_executing = True

        if re_executing:
            self.logger.info("Going to re-execute karma code")
            self.exec_dir.mkdir(exist_ok=True)
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))

            setup_karma(self.dataset, self.exec_dir, self.sm_type_dir)
            execute_karma_code(self.dataset, self.exec_dir, self.use_correct_type, self.use_old_semantic_typer,
                               self.num_candidate_semantic_type,
                               self.multiple_same_property_per_node, self.coherence, self.confidence,
                               self.size_reduction, self.num_candidate_mappings, self.mapping_branching_factor,
                               self.topk_steiner_tree, self.cut_off, train_source_names, test_source_names)

            # only have this file when previous execution is success!
            with open(execution_meta_file, 'w') as f:
                ujson.dump(self.get_meta(train_source_names, test_source_names), f, indent=4)

        self.__has_run_modeling = True

    def karma_model_candidate_generation(self,
                                         train_sms: List[SemanticModel],
                                         test_sms: List[SemanticModel],
                                         n_candidate: int = 1000) -> List[List[KarmaModel]]:
        if not self.__has_run_modeling:
            self.init([s.id for s in train_sms], [s.id for s in test_sms])

        self.logger.debug("Load previous result...")
        results = []
        karma_models_dir = Path(config.datasets[self.dataset].karma_version.as_path()) / "models-json"
        for test_sm in test_sms:
            file_name = "source--%s.json" % test_sm.id
            predicted_models: List[KarmaModel] = []

            if self.use_old_semantic_typer:
                karma_sm = KarmaModel.load_from_file(self.ont,
                                                     self.exec_dir / "output" / f"source--{test_sm.id}.original.json")
            else:
                karma_sm = KarmaModel.load_from_file(self.ont, karma_models_dir / f"{test_sm.id}-model.json")
            sm_alignment: SemanticModelAlignment = SemanticModelAlignment(test_sm, karma_sm)

            with open(self.exec_dir / "output" / file_name, 'r') as f:
                for i, serialized_sm in enumerate(f):
                    pred_sm = sm_alignment.load_and_align(self.ont, serialized_sm)
                    pred_sm.id = f"{test_sm.id}:::{i}"
                    predicted_models.append(pred_sm)
                    if (i + 1) >= n_candidate:
                        break

            if len(predicted_models) == 0:
                karma_graph = KarmaGraph(True, True, True)
                for dnode in karma_sm.karma_graph.iter_data_nodes():
                    karma_graph.real_add_new_node(KarmaGraphNode([], [], dnode.literal_type, dnode.is_literal_node),
                                                  GraphNodeType.DATA_NODE, dnode.label)
                karma_model = KarmaModel(karma_sm.id, karma_sm.description, karma_sm.source_columns,
                                         karma_sm.mapping_to_source_columns, karma_graph)
                predicted_models = [karma_model]

            assert len(predicted_models) == len({m.id for m in predicted_models}), "No id duplication"
            results.append(predicted_models)

        return results

    def sm_candidate_generation(self, training_sources: List[SemanticModel],
                                testing_sources: List[SemanticModel]) -> List[List[SemanticModel]]:
        results = self.karma_model_candidate_generation(training_sources, testing_sources)
        return [[m.get_semantic_model() for m in predicted_models] for predicted_models in results]

    def semantic_labeling(self, training_sources: List[SemanticModel], testing_sources: List[SemanticModel]) -> List[
        Dict[str, List[KarmaSemanticType]]]:
        """This method perform """
        results = self.karma_model_candidate_generation(training_sources, testing_sources, n_candidate=1)
        node2stypes = []

        for test_sm, predicted_models in zip(testing_sources, results):
            node2stypes.append({node.label.decode("utf-8"): node.learned_semantic_types for node in
                                predicted_models[0].karma_graph.iter_data_nodes()})

        return node2stypes

    def sm_prediction(self, training_sources: List[SemanticModel], testing_sources: List[SemanticModel]) -> List[
        SemanticModel]:
        return [pred_sms[0] for pred_sms in self.sm_candidate_generation(training_sources, testing_sources)]
