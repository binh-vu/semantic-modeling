#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import chain
from typing import Dict, Tuple, List, Union, Optional, Iterable

import torch

from gmtk.graph_models.factors.factor import DotTensor1Factor, Factor
from gmtk.graph_models.factors.template_factor import CachedTemplateFactorConstructor
from gmtk.graph_models.models.weights import Weights
from gmtk.graph_models.variables.utils import iter_assignment
from gmtk.graph_models.variables.vector_domain import BinaryVectorValue
from gmtk.inferences.inference import Inference
from gmtk.tensors import DenseTensor, DenseTensorFunc
from semantic_modeling.assembling.undirected_graphical_model.model_core import TripleLabel


class ExternalModelFactorTemplate(CachedTemplateFactorConstructor[TripleLabel]):
    class MixedExternalModelFactor(DotTensor1Factor[TripleLabel, BinaryVectorValue]):
        def __init__(self, variables: List[TripleLabel], weights: Weights, external_weights: Weights) -> None:
            super().__init__(variables, weights)
            self.triple = variables[0].triple
            source_features = self.triple.example.node2features[self.triple.link.source_id]

            self.exmodel_input_s = self._get_exmodel_input(source_features)
            if self.triple.link.get_target_node().is_class_node():
                target_features = self.triple.example.node2features[self.triple.link.target_id]
                self.exmodel_input_d = self._get_exmodel_input(target_features)
            else:
                self.exmodel_input_d = None
            self.external_weights = external_weights

        def after_update_weights(self):
            self._features_tensor = None

        def val2features(self, vals: List[BinaryVectorValue]) -> DenseTensor:
            return vals[0].tensor * self._get_exmodel_output()

        def unobserved_val2features(self, vals: List[BinaryVectorValue]) -> DenseTensor:
            return vals[0].tensor * self._get_exmodel_output()

        def compute_gradients(self, target_assignment: Dict[TripleLabel, BinaryVectorValue],
                              inference: Inference) -> Iterable[Tuple[DenseTensor, DenseTensor]]:
            gradients = super().compute_gradients(target_assignment, inference)
            # new autograd
            weights_var = DenseTensorFunc.tensor2torch_var(self.weights.val)
            external_weights_var = DenseTensorFunc.tensor2torch_var(self.external_weights.val, requires_grad=True)

            features = self.triple.example.link2features[self.triple.link.id]
            p_source = torch.sigmoid(
                torch.dot(external_weights_var, DenseTensorFunc.tensor2torch_var(self.exmodel_input_s)))
            if self.exmodel_input_d is not None:
                p_target = torch.sigmoid(
                    torch.dot(external_weights_var, DenseTensorFunc.tensor2torch_var(self.exmodel_input_d)))
            else:
                p_target = 1
            p_link_given_so = features['p_link_given_so']
            prob_triple = p_source * p_target * p_link_given_so
            ex_output = torch.stack([1 - prob_triple, prob_triple]).view(-1)

            w_grads = [None, None]
            for assigned_indice in iter_assignment(self.unobserved_variables):
                val = DenseTensorFunc.tensor2torch_var(self.variables[0].get_value().tensor)
                output = torch.dot(val * ex_output, weights_var)
                # eq_(output.size()[0], 1)
                # eq_(output.data[0], self.current_score())

                output.backward(retain_graph=True)
                w_grads[assigned_indice[0]] = DenseTensorFunc.from_numpy_array(external_weights_var.grad.data.numpy())
                external_weights_var.grad.zero_()

            w_grads = DenseTensorFunc.stack(w_grads)  # Y x W
            target_idx = target_assignment[self.variables[0]].idx
            exw_grad = w_grads[target_idx, :] - inference.log_prob_factor(self).exp().view(1, -1).mm(w_grads).squeeze()
            return chain(gradients, [(self.external_weights, exw_grad)])

        def _get_exmodel_output(self) -> DenseTensor:
            """Compute output that will go to dot product to get score of a factor"""
            features = self.triple.example.link2features[self.triple.link.id]
            p_source = self.external_weights.val.dot(self.exmodel_input_s).sigmoid_()

            if self.exmodel_input_d is not None:
                p_target = self.external_weights.val.dot(self.exmodel_input_d).sigmoid_()
            else:
                # TODO: hard-code prob. of target nodes
                p_target = 1
            p_link_given_so = features['p_link_given_so']

            prob_y = p_source * p_target * p_link_given_so
            return DenseTensorFunc.stack([1 - prob_y, prob_y])

        def _get_exmodel_input(self, node_features) -> DenseTensor:
            # noinspection PyProtectedMember
            return DenseTensorFunc.from_array(
                [node_features['prob_data_nodes'], node_features['minimum_merged_cost'], 1])

    def __init__(self, weights: DenseTensor, external_weights: DenseTensor) -> None:
        super().__init__(weights)
        self.external_weights = Weights(external_weights)

    def after_update_weights(self):
        assert len(self.cached_factors) > 0
        for factors in self.cached_factors.values():
            if len(factors) > 0:
                factors[0].after_update_weights()

    @staticmethod
    def get_default_weights():
        # TODO: init args properly
        return DenseTensorFunc.create_randn((2, )), DenseTensorFunc.create_randn((3, ))

    def get_weights(self):
        return [self.weights, self.external_weights]

    def cached_unroll(self, var: TripleLabel) -> List[Factor]:
        if var.triple.target.is_data_node():
            return []
        return [ExternalModelFactorTemplate.MixedExternalModelFactor([var], self.weights, self.external_weights)]