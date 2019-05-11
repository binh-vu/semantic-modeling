#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import combinations, islice, chain
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from data_structure import GraphNode
from gmtk.graph_models.factors.factor_group import GroupedTensorFactor, SubTensorFactor
from gmtk.graph_models.factors.template_factor import TemplateFactorConstructor
from gmtk.graph_models.models.weights import Weights
from gmtk.graph_models.variables.vector_domain import GrowableBinaryVectorDomain, BinaryVectorValue, BooleanVectorDomain
from gmtk.tensors import DenseTensor, DenseTensorFunc
from semantic_modeling.assembling.undirected_graphical_model.factors.duplication_factor import DuplicationFactor
from semantic_modeling.assembling.undirected_graphical_model.factors.local_structure_factor import \
    AllChildrenWrongFactor
from semantic_modeling.assembling.undirected_graphical_model.factors.pairwise_factor import PairwisePrimaryKeyFactor, \
    PairwiseAttributeScopeFactor
from semantic_modeling.assembling.undirected_graphical_model.model_core import TripleLabel, ExampleAnnotator, Triple
from semantic_modeling.assembling.weak_models.attribute_scope import AttributeScope
from semantic_modeling.assembling.weak_models.cardinality_matrix import CardinalityFeatures
from semantic_modeling.assembling.weak_models.local_structure import LocalStructure
from semantic_modeling.assembling.weak_models.primary_key import PrimaryKey
from semantic_modeling.assembling.weak_models.structures.duplication_tensors import DuplicationTensors
from semantic_modeling.settings import Settings


class SubstructureFactorTemplate(TemplateFactorConstructor[TripleLabel]):
    """VariableID != VariableIndex"""

    def __init__(self, all_children_weights: DenseTensor, pairwise_pk_weights: DenseTensor, pairwise_scope_weights: DenseTensor,
                 duplication_weights: Dict[str, DenseTensor], pairwise_domain: GrowableBinaryVectorDomain[str]) -> None:
        self.settings = Settings.get_instance()

        self.all_children_weights: Weights = Weights(all_children_weights)
        self.pairwise_pk_weights: Weights = Weights(pairwise_pk_weights)
        self.pairwise_scope_weights: Weights = Weights(pairwise_scope_weights)
        self.duplication_weights: Dict[str, Weights] = {k: Weights(v) for k, v in duplication_weights.items()}

        self.pairwise_domain = pairwise_domain
        self.boolean_domain = BooleanVectorDomain.get_instance()
        # use to compute pairwise factor's feature tensor
        # similar to DotTensor1WithSufficientStatisticFactor.get_feature_tensor#domain_tensor
        self.pairwise_indice_func_tensor = DenseTensorFunc.from_array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).view(4, 4, 1)

        # # use to set local structure correctly (not duplications)
        # self.dup_assigning_values: List[List[List[int]]] = [None]
        # for i in range(1, self.max_n_duplications + 1):
        #     assigning_values = [[1] * i]
        #     for n_zero in range(1, i):
        #         for indices in combinations(range(i), n_zero):
        #             val = [1] * i
        #             for idx in indices:
        #                 val[idx] = 0
        #             assigning_values.append(val)
        #
        #     assigning_values.append([0] * i)
        #     self.dup_assigning_values.append(assigning_values)

    def get_weights(self):
        weights = [self.all_children_weights, self.pairwise_pk_weights, self.pairwise_scope_weights]
        for w in self.duplication_weights.values():
            weights.append(w)
        return weights

    def get_info(self):
        return "AllChildrenWrongFactor,PairwisePrimaryKeyFactor,PairwiseScopeFactor,DuplicationFactor"

    @staticmethod
    def get_default_args(pairwise_domain: GrowableBinaryVectorDomain, obj_props: List[Tuple[bytes, bytes]]):
        return DenseTensorFunc.zeros((2, )), \
               DenseTensorFunc.zeros((4 * pairwise_domain.size()[0], )), \
               DenseTensorFunc.zeros((4 * 2, )), \
               {obj_prop: DenseTensorFunc.zeros((2, )) for obj_prop in obj_props}, pairwise_domain

    def unroll(self, var: TripleLabel) -> List[SubTensorFactor[TripleLabel, BinaryVectorValue]]:
        if len(var.triple.children) == 0 and not var.triple.is_root_triple:
            return []

        annotator: ExampleAnnotator = var.triple.example.annotator
        local_structure: LocalStructure = annotator.local_structure

        grouped_factors: List[SubTensorFactor] = []
        if var.triple.is_root_triple and len(var.triple.siblings) > 0:
            child_vars = var.triple.siblings + [var.triple]
            variables, var_ids = self._prepare_args(local_structure, None, child_vars)

            factors = self.create_pairwise_factors(variables, None, annotator.cardinality, child_vars[0].source.label, annotator.primary_key)
            factors += self.create_pairwise_scope_factors(variables, 0, annotator.attribute_same_scope)
            factors += self.create_duplication_factors(variables, var_ids, child_vars[0].source.label, annotator.duplication_tensors)

            grouped_factors.append(GroupedTensorFactor(variables, factors))

        if len(var.triple.children) > 1:
            child_vars = var.triple.children
            variables, var_ids = self._prepare_args(local_structure, var.triple, child_vars)

            factors = self.create_pairwise_factors(variables, var, annotator.cardinality, child_vars[0].source.label, annotator.primary_key)
            factors += self.create_pairwise_scope_factors(variables, 1, annotator.attribute_same_scope)
            factors += self.create_duplication_factors(variables, var_ids, child_vars[0].source.label,
                                                                  annotator.duplication_tensors)
            factors.append(AllChildrenWrongFactor(variables, self.all_children_weights))
            grouped_factors.append(GroupedTensorFactor(variables, factors))

        return grouped_factors

    def create_pairwise_factors(self, variables: List[TripleLabel], ignore_var: Optional[TripleLabel],
                                cardinality_feature: CardinalityFeatures, source_lbl: bytes, primary_keys: PrimaryKey):
        if source_lbl not in primary_keys:
            return []

        pk = primary_keys[source_lbl]
        idx = next((i for i, x in enumerate(variables) if x.triple.link.label == pk), -1)
        if idx == -1:
            return []

        factors = []
        source_lbl = source_lbl.decode('utf-8')
        pk = pk.decode('utf-8')
        sm_id = variables[0].triple.example.get_model_id()
        pk_col = variables[idx].triple.target.label

        if ignore_var is not None:
            iter_vars = enumerate(islice(variables, 1, None), 1)
        else:
            iter_vars = enumerate(variables)

        for i, x in iter_vars:
            if idx != i:
                x_target = x.triple.target
                if x_target.is_data_node():
                    cardin = cardinality_feature.get_cardinality(sm_id, pk_col, x.triple.target.label)
                else:
                    # if x_target is class node, we will compare between 2 primary keys
                    # TODO: fix me, here we assume the primary keys of next class is correct, but it may not
                    dpk = primary_keys[x_target.label]
                    e = next((e for e in x_target.iter_outgoing_links() if e.label == dpk), None)
                    if e is None:
                        continue
                    cardin = cardinality_feature.get_cardinality(sm_id, pk_col, e.get_target_node().label)

                cat = f"source={source_lbl},x={pk},y={x.triple.link.label.decode('utf-8')},cardinality={cardin}"
                if self.pairwise_domain.has_value(cat):
                    feature_tensor = self.pairwise_indice_func_tensor.matmul(self.pairwise_domain.encode_value(cat).tensor.view(1, 1, -1)).view(4, -1)
                    input_var_idx = [idx, i] if idx <= i else [i, idx]
                    factors.append(PairwisePrimaryKeyFactor(variables, self.pairwise_pk_weights, input_var_idx, feature_tensor))

        return factors

    def create_pairwise_scope_factors(self, variables: List[TripleLabel], start_idx: int, attribute_scope: AttributeScope):
        sm_id = variables[0].triple.example.get_model_id()
        factors = []

        for i in range(start_idx, len(variables)):
            x_target = variables[i].triple.target
            if x_target.is_class_node():
                continue

            for j in range(i + 1, len(variables)):
                y_target = variables[j].triple.target
                if y_target.is_data_node():
                    val = self.boolean_domain.encode_value(attribute_scope.is_same_scope(sm_id, x_target.label, y_target.label))
                    feature_tensor = self.pairwise_indice_func_tensor.matmul(val.tensor.view(1, 1, -1)).view(4, -1)
                    input_var_idx = [i, j]
                    factors.append(PairwiseAttributeScopeFactor(variables, self.pairwise_scope_weights, input_var_idx, feature_tensor))

        return factors

    def create_duplication_factors(self, variables: List[TripleLabel], var_ids: List[int], source_lbl: bytes,
                                   duplication_tensors: DuplicationTensors):
        factors = []
        dup_range = [0, 1]

        for i, id in enumerate(islice(var_ids, 1, None), 1):
            if var_ids[i - 1] == id:
                dup_range[1] += 1
            else:
                if dup_range[1] - dup_range[0] > 1:
                    weights = self.duplication_weights[(source_lbl, variables[i-1].triple.link.label)]
                    tensor = duplication_tensors.get_tensor(source_lbl, var_ids[i - 1], dup_range[1] - dup_range[0])
                    factors.append(DuplicationFactor(variables, weights, list(range(*dup_range)), tensor))
                dup_range = [i, i + 1]

        if dup_range[1] - dup_range[0] > 1:
            weights = self.duplication_weights[(source_lbl, variables[i - 1].triple.link.label)]
            tensor = duplication_tensors.get_tensor(source_lbl, var_ids[i - 1], dup_range[1] - dup_range[0])
            factors.append(DuplicationFactor(variables, weights, list(range(*dup_range)), tensor))

        return factors

    def _prepare_args(self, local_structure: LocalStructure, parent_var: Optional[Triple], child_vars: List[Triple]):
        node: GraphNode = child_vars[0].source
        if node.label not in local_structure.node_structure_space:
            # only appear in test
            return None, None

        node_structure = local_structure.node_structure_space[node.label]

        if parent_var is None:
            parent_var_with_ids = []
        else:
            parent = (parent_var.link.label, parent_var.source.label)
            parent_id = node_structure.get_parent_idx(parent[0], parent[1])  # can be None
            parent_var_with_ids = [(parent_var, parent_id)]

        child_var_with_ids = (
            (var, node_structure.get_child_idx(var.link.label,
                                               b"DATA_NODE" if var.target.is_data_node() else var.target.label))
            for var in child_vars)
        var_with_ids = [x for x in chain(parent_var_with_ids, child_var_with_ids) if x[1] is not None]
        var_with_ids.sort(key=lambda x: x[1])  # lower index, meaning higher frequency

        if len(var_with_ids) > self.settings.mrf_max_n_props:
            # only select top-k vars
            counter = 0
            for var, idx in var_with_ids:
                counter += 1
                if var.target.is_class_node():
                    # must include class nodes
                    continue

                if counter > self.settings.mrf_max_n_props:
                    break
            var_with_ids = var_with_ids[:counter]

        variables: List[TripleLabel] = []
        var_ids: List[int] = []
        for var, idx in var_with_ids:
            variables.append(var.label)
            var_ids.append(idx)

        return variables, var_ids