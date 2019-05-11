#!/usr/bin/python
# -*- coding: utf-8 -*-
import multiprocessing
from pathlib import Path
from typing import Dict, List

import numpy as np
from pyutils.list_utils import _

from gmtk.graph_models.factors.template_factor import CachedTemplateFactorConstructor
from gmtk.graph_models.models.model import TemplateLogLinearModel
from gmtk.graph_models.variables.vector_domain import GrowableBinaryVectorDomain
from gmtk.inferences import BeliefPropagation, parallel_marginal_inference
from gmtk.inferences.inference import InferProb
from gmtk.optimize.example import MAPAssignmentExample
from gmtk.tensors import DenseTensorFunc
from semantic_modeling.assembling.cshare.merge_graph import MergeGraph
from semantic_modeling.settings import Settings
from semantic_modeling.assembling.learning.shared_models import Example
from semantic_modeling.assembling.undirected_graphical_model.model_core import ExampleAnnotator
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import deserialize


class Model(object):
    def __init__(self, dataset: str, model: TemplateLogLinearModel, tf_domain: GrowableBinaryVectorDomain, pairwise_domain: GrowableBinaryVectorDomain) -> None:
        self.dataset = dataset
        self.source_models: Dict[str, SemanticModel] = {s.id: s for s in get_semantic_models(dataset)}
        self.inference = BeliefPropagation.get_constructor(InferProb.MARGINAL)
        self.map_inference = BeliefPropagation.get_constructor(InferProb.MAP)
        self.model: TemplateLogLinearModel = model
        for template in model.templates:
            if isinstance(template, CachedTemplateFactorConstructor):
                template.disable_cache()
        self.tf_domain: GrowableBinaryVectorDomain = tf_domain
        self.pairwise_domain = pairwise_domain
        self.example_annotator: ExampleAnnotator = None
        self.max_n_tasks = Settings.get_instance().max_n_tasks

    @staticmethod
    def from_file(dataset: str, input_dir: Path) -> 'Model':
        model_bin = deserialize(input_dir / 'gmtk_model.bin')
        model: TemplateLogLinearModel = model_bin[0]
        tf_domain: GrowableBinaryVectorDomain = model_bin[1]
        pairwise_domain = model_bin[2]
        return Model(dataset, model, tf_domain, pairwise_domain)

    def update_tensor_types_(self) -> "Model":
        """in a way that we won't have any problems with TensorC and TensorPy"""
        for param in self.model.get_parameters():
            param.val = DenseTensorFunc.from_numpy_array(param.val.numpy())

        if self.tf_domain._domain_tensor is not None:
            self.tf_domain._domain_tensor = DenseTensorFunc.from_numpy_array(self.tf_domain._domain_tensor.numpy())
        if self.pairwise_domain._domain_tensor is not None:
            self.pairwise_domain._domain_tensor = DenseTensorFunc.from_numpy_array(self.pairwise_domain._domain_tensor.numpy())
        return self

    def get_map_example(self, example: Example) -> MAPAssignmentExample:
        vars = self.get_variables(example)
        map_example = MAPAssignmentExample(vars, self.model.get_factors(vars), self.map_inference)
        return map_example

    def predict_link2label(self, example: Example) -> Dict[int, bool]:
        map = self.get_map_example(example).get_map_assignment()
        return {v.triple.link.id: x.val for v, x in map.items()}

    def predict_log_prob(self, example: Example):
        vars = self.get_variables(example)
        factors = self.model.get_factors(vars)
        inference = self.inference(factors, vars)
        desired_assignment = {var: var.domain.encode_value(True) for var in vars}

        logZ = inference.logZ()
        log_prob = sum(f.score_assignment(desired_assignment) for f in factors) - logZ
        return log_prob

    # @profile
    def predict_log_probs(self, examples: List[Example]):
        log_probs = []

        for es in _(examples).isplit(self.max_n_tasks):
            varss = [self.get_variables(e) for e in es]
            # varss = self.parallel_get_variables(es)
            factorss = [self.model.get_factors(vars) for vars in varss]
            inferences = [self.inference(f, v) for f, v in zip(factorss, varss)]

            desired_assignments = [
                {var: var.domain.encode_value(True) for var in vars}
                for vars in varss
            ]

            logZs = parallel_marginal_inference(inferences, n_threads=Settings.get_instance().parallel_gmtk_n_threads)
            log_probs += [
                sum(f.score_assignment(desired_assignments[i]) for f in factorss[i]) - logZs[i]
                for i in range(len(es))
            ]

        return log_probs

    def predict_prob(self, example: Example):
        return np.exp(self.predict_log_prob(example))

    # @profile
    def predict_probs(self, examples: List[Example]):
        # return [self.predict_prob(example) for example in examples]
        return [np.exp(x) for x in self.predict_log_probs(examples)]

    def predict_sm_probs(self, sm_id: str, train_sm_ids: List[str],  gs: List[MergeGraph]):
        examples = [
            Example(None, g, {
                link.id: True for link in g.iter_links()
            }, None)
            for g in gs
        ]
        for example in examples:
            example.set_meta(example.generate_example_id(sm_id, 0, 0), train_sm_ids)

        return self.predict_probs(examples)

    def get_variables(self, example: Example):
        if self.example_annotator is None:
            # TODO: should handle top_k_semantic_types configuration, and check if training_sources has been changed!!
            self.example_annotator = ExampleAnnotator(self.dataset, example.training_sources)
        self.example_annotator.annotate(example)

        return _(self.example_annotator.example2vars(example)) \
            .imap(lambda x: self.example_annotator.build_triple_features(x, self.tf_domain)) \
            .map(lambda x: x.label)

    def parallel_get_variables(self, examples: List[Example]):
        assert False, "Not ready to use yet"
        ParallelAnnotator.get_instance().annotate(examples)
        return [
            _(self.example_annotator.example2vars(example)) \
                .imap(lambda x: self.example_annotator.build_triple_features(x, self.tf_domain)) \
                .map(lambda x: x.label)
            for example in examples
        ]

    def __getstate__(self):
        return self.source_models, self.model, self.tf_domain, self.pairwise_domain

    def __setstate__(self, state):
        self.source_models = state[0]
        self.model = state[1]
        self.tf_domain = state[2]
        self.pairwise_domain = state[3]
        self.inference = BeliefPropagation.get_constructor(InferProb.MARGINAL)
        self.map_inference = BeliefPropagation.get_constructor(InferProb.MAP)


class ParallelAnnotator(object):

    instance = None

    def __init__(self, dataset: str, train_source_ids: List[str]) -> None:
        self.n_processes = Settings.get_instance().parallel_n_annotators
        self.annotator = ExampleAnnotator(dataset, train_source_ids)

        self.processes = []
        for i in range(self.n_processes):
            parent_conn, child_conn = multiprocessing.Pipe()
            process = multiprocessing.Process(
                target=ParallelAnnotator.parallel_annotate,
                name=f"parallel-annotator-{i}", args=(child_conn,))
            self.processes.append({
                "parent_conn": parent_conn,
                "child_conn": child_conn,
                "process": process
            })
            process.start()

        for proc_info in self.processes:
            proc_info['parent_conn'].send({
                "message": "start",
                "dataset": dataset,
                "train_sm_ids": train_source_ids
            })

    @staticmethod
    def get_instance(dataset: str=None, train_source_ids: List[str]=None):
        if ParallelAnnotator.instance is None:
            ParallelAnnotator.instance = ParallelAnnotator(dataset, train_source_ids)
        return ParallelAnnotator.instance

    def stop(self):
        for process in self.processes:
            process['parent_conn'].send({"message": "stop"})

    def annotate(self, examples: List[Example]):
        for i, example in enumerate(examples):
            self.processes[i % self.n_processes]['parent_conn'].send({
                "message": "annotate",
                "example": example
            })

        for i, example in enumerate(examples):
            result = self.processes[i % self.n_processes]['parent_conn'].recv()
            example.node2features = result['node2features']
            example.link2features = result['link2features']

        return examples

    @staticmethod
    def parallel_annotate(conn: 'multiprocessing.Connection'):
        message = conn.recv()
        assert message['message'] == 'start'
        example_annotator = ExampleAnnotator(message['dataset'], message['train_sm_ids'])

        while True:
            message = conn.recv()
            if message['message'] == "stop":
                break

            assert message['message'] == 'annotate'
            example: Example = message['example']
            example_annotator.annotate(example)

            conn.send({
                "node2features": example.node2features,
                "link2features": example.link2features
            })