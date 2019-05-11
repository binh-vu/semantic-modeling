#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any

import numpy

from gmtk.graph_models.factors.factor_group import SubTensorFactor
from gmtk.graph_models.models.weights import Weights
from gmtk.graph_models.variables.vector_domain import BinaryVectorValue
from gmtk.tensors import DenseTensor, DenseTensorFunc
from semantic_modeling.assembling.undirected_graphical_model.model_core import TripleLabel


class AllChildrenWrongFactor(SubTensorFactor[TripleLabel, BinaryVectorValue]):
    """Take list of variables belongs to a class node (incoming links and outgoing links) as `variables`,
    The first variable is always incoming link, and the rest are outgoing links.
    """

    def __init__(self, variables: List[TripleLabel], weights: Weights) -> None:
        super().__init__(variables, weights)
        self.real_var_dims = [2] * (len(self.variables) + 1)
        self.real_var_dims[-1] = self.weights.val.size()[0]

    def score(self, vars: List[TripleLabel]) -> float:
        assigned_indice = [var.get_value().idx for var in vars]
        target_idx = numpy.ravel_multi_index(assigned_indice, self.vars_dims)
        features = self._features_tensor[target_idx, :]

        return features.dot(self.weights.val).get_scalar()

    def score_assignment(self, assignment: Dict[TripleLabel, BinaryVectorValue]) -> float:
        assigned_indice = [assignment[var].idx for var in self.unobserved_variables]
        target_idx = numpy.ravel_multi_index(assigned_indice, self.vars_dims)
        features = self._features_tensor[target_idx, :]

        return features.dot(self.weights.val).get_scalar()

    def get_scores_tensor(self) -> DenseTensor:
        if self._features_tensor is None:
            self._features_tensor = DenseTensorFunc.zeros((2, 2 ** (len(self.variables) - 1), self.real_var_dims[-1]))
            self._features_tensor[0, 0, 0] = 1
            self._features_tensor[1, 0, 1] = 1
            self._features_tensor = self._features_tensor.view(-1, self.real_var_dims[-1])

        return self._features_tensor.mv(self.weights.val).view_shape(self.real_var_dims[:-1])

    def compute_gradients(self, target_assignment: Dict[TripleLabel, BinaryVectorValue],
                          prob_factor: DenseTensor):
        assigned_indice = [target_assignment[var].idx for var in self.unobserved_variables]
        target_idx = numpy.ravel_multi_index(assigned_indice, self.vars_dims)
        expectation = prob_factor.view(1, -1).mm(self._features_tensor).squeeze()
        return [(self.weights, self._features_tensor[target_idx, :] - expectation)]

    def assignment2features(self, assignment: Dict[TripleLabel, BinaryVectorValue]) -> float:
        assigned_indice = [assignment[var].idx for var in self.unobserved_variables]
        target_idx = numpy.ravel_multi_index(assigned_indice, self.vars_dims)
        return self._features_tensor[target_idx, :]