#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy, numpy as np
from typing import Dict, Tuple, List, Set, Union, Optional, Iterable

from pyutils.list_utils import _

from gmtk.graph_models.factors.factor import DotTensor1Factor, IDiscreteVar, IDiscreteVal, Tensor
from gmtk.graph_models.factors.factor_group import SubTensorFactor
from gmtk.graph_models.factors.template_factor import TemplateFactorConstructor
from gmtk.graph_models.models.weights import Weights
from gmtk.graph_models.variables.vector_domain import BinaryVectorValue
from gmtk.tensors import DenseTensor, DenseTensorFunc
from semantic_modeling.assembling.undirected_graphical_model.model_core import TripleLabel


class DuplicationFactor(SubTensorFactor[TripleLabel, BinaryVectorValue]):
    """Take list of variables belongs to a class node (incoming links and outgoing links) as `variables`,
    and a list of variable index `input_var_idx` point to index of two variables that are actual input to the factor
    """

    def __init__(self, variables: List[TripleLabel], weights: Weights, input_var_idx: List[int], dup_tensor: DenseTensor) -> None:
        super().__init__(variables, weights)
        self.input_var_idx = input_var_idx
        self.real_var_dims = [1] * (len(self.vars_dims) + 1)
        for idx in input_var_idx:
            self.real_var_dims[idx] = self.vars_dims[idx]
        self.real_var_dims[-1] = self.weights.val.size()[0]
        self.full_var_dims = [2] * (len(self.vars_dims) + 1)
        self.full_var_dims[-1] = self.real_var_dims[-1]
        self.sub_var_dims = [2] * len(input_var_idx)

        # dup_tensor is equal to features_tensor
        self._features_tensor = dup_tensor.view(-1, self.real_var_dims[-1])

    def score(self, vars: List[TripleLabel]) -> float:
        assigned_indice = [int(vars[idx].get_value().val) for idx in self.input_var_idx]
        target_idx = np.ravel_multi_index(assigned_indice, self.sub_var_dims)
        features = self._features_tensor[target_idx, :]

        return features.dot(self.weights.val).get_scalar()

    def score_assignment(self, assignment: Dict[TripleLabel, BinaryVectorValue]) -> float:
        assigned_indice = [int(assignment[self.variables[idx]].val) for idx in self.input_var_idx]
        target_idx = np.ravel_multi_index(assigned_indice, self.sub_var_dims)
        features = self._features_tensor[target_idx, :]

        return features.dot(self.weights.val).get_scalar()

    def get_scores_tensor(self) -> DenseTensor:
        return self._features_tensor.mv(self.weights.val).view_shape(self.real_var_dims[:-1])

    def compute_gradients(self, target_assignment: Dict[IDiscreteVar, IDiscreteVal],
                          prob_factor: DenseTensor):
        # TODO: can improve this function
        prob_factor_shape = list(prob_factor.size())
        prob_factor_shape.append(1)
        sub_prob_factor = prob_factor.view_shape(prob_factor_shape)

        expectation = (self._features_tensor.view_shape(self.real_var_dims) * sub_prob_factor).view(-1, self.real_var_dims[-1]).sum(0)
        assigned_indice = [int(target_assignment[self.variables[idx]].val) for idx in self.input_var_idx]
        target_idx = np.ravel_multi_index(assigned_indice, self.sub_var_dims)

        return [(self.weights, self._features_tensor[target_idx, :] - expectation)]

    def assignment2features(self, assignment: Dict[TripleLabel, BinaryVectorValue]) -> float:
        assigned_indice = [int(assignment[self.variables[idx]].val) for idx in self.input_var_idx]
        target_idx = np.ravel_multi_index(assigned_indice, self.sub_var_dims)
        return self._features_tensor[target_idx, :]