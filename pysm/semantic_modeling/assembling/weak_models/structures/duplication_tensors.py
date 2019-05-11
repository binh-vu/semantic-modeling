#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, List

from gmtk.tensors import DenseTensor, DenseTensorFunc
from semantic_modeling.assembling.weak_models.local_structure import LocalStructure
from semantic_modeling.assembling.weak_models.multi_val_predicate import MultiValuePredicate
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.settings import Settings


class DuplicationTensors(object):

    instance = None

    def __init__(self, multi_val_predicate: MultiValuePredicate, structure: LocalStructure):
        self.tensors: Dict[bytes, Dict[int, List[DenseTensor]]] = {}
        features = DenseTensorFunc.from_array([0, 0])
        n_features = features.size()[0]

        max_n_dups = Settings.get_instance().mrf_max_n_duplications

        for lbl, space in structure.node_structure_space.items():
            self.tensors[lbl] = {}
            for ctriple, child_idx in space.children.items():
                self.tensors[lbl][child_idx] = [None, None]
                for n_dup in range(2, max_n_dups + 1):
                    tensor = DenseTensorFunc.zeros((2 ** n_dup, n_features))
                    dims = [2] * n_dup
                    dims.append(n_features)
                    for count, current_val_index, values in iter_values(n_dup, 0):
                        if len(values) <= 1:
                            features[0] = 0
                            features[1] = 0
                        else:
                            multi_val_prob = multi_val_predicate.compute_prob(ctriple[0], len(values))
                            features[0] = max(multi_val_prob, 0.01)
                            features[1] = max(1 - multi_val_prob, 0.01)

                        tensor[count, :] = features

                    self.tensors[lbl][child_idx].append(tensor.view_shape(dims))

    def get_tensor(self, source_lbl: bytes, link_id: int, n_dups: int) -> DenseTensor:
        return self.tensors[source_lbl][link_id][n_dups]

    @staticmethod
    def get_instance(train_sms: List[SemanticModel]):
        if DuplicationTensors.instance is None:
            DuplicationTensors.instance = DuplicationTensors(MultiValuePredicate.get_instance(train_sms), LocalStructure.get_instance(train_sms))

        return DuplicationTensors.instance


def iter_values(n_children: int, n_offset):
    """Copy & modify from gmtk.utils"""
    current_val_index = [0] * n_children
    max_val_index = [2] * n_children
    values = set()
    count = 0

    yield count, current_val_index, values

    while True:
        # iterate through each assignment
        i = len(current_val_index) - 1
        while i >= 0:
            # move to next state & set value of variables to next state value
            current_val_index[i] += 1
            if current_val_index[i] == max_val_index[i]:
                current_val_index[i] = 0
                values.remove(i + n_offset)
                i = i - 1
            else:
                values.add(i + n_offset)
                break

        if i < 0:
            # already iterated through all values
            break

        count += 1
        # yield current values
        yield count, current_val_index, values


