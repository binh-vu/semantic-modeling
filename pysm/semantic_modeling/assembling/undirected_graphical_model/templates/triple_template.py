#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any

from gmtk.graph_models.factors.factor import Factor
from gmtk.graph_models.factors.template_factor import CachedTemplateFactorConstructor
from gmtk.graph_models.factors.typical_factor import DotTensor1WithSufficientStatisticFactor
from gmtk.graph_models.variables.vector_domain import BinaryVectorValue, GrowableBinaryVectorDomain
from gmtk.tensors import DenseTensorFunc
from semantic_modeling.assembling.undirected_graphical_model.model_core import TripleLabel


class TripleFactor(DotTensor1WithSufficientStatisticFactor[TripleLabel, BinaryVectorValue]):
    pass


class TripleFactorTemplate(CachedTemplateFactorConstructor[TripleLabel]):

    def cached_unroll(self, var: TripleLabel) -> List[Factor]:
        return [TripleFactor([var, var.triple.features], self.weights)]

    @staticmethod
    def get_default_args(tf_domain: GrowableBinaryVectorDomain[str]):
        return DenseTensorFunc.create_randn((2, tf_domain.size()[0])).view(-1).contiguous_(),