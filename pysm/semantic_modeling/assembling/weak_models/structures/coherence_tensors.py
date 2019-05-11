#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any

from gmtk.tensors import DenseTensor, DenseTensorFunc
from semantic_modeling.assembling.weak_models.local_structure import LocalStructure
from semantic_modeling.assembling.weak_models.structures.duplication_tensors import iter_values
from semantic_modeling.karma.semantic_model import SemanticModel


class CoherenceTensors:

    instance = None

    def __init__(self, structure: LocalStructure, train_sms: List[SemanticModel]):
        self.tensors: Dict[bytes, Dict[Optional[int], DenseTensor]] = {}
        substructure = {}

        # PRE-PROCESSING FOR BUILDING COHERENCE
        for sm in train_sms:
            for n in sm.graph.iter_class_nodes():
                local_structure = set()
                plink = n.get_first_incoming_link()
                if plink is not None:
                    pindex = structure.node_structure_space[n.label].parents[(plink.label, plink.get_source_node().label)]
                    local_structure.add(pindex)

                children_offset = len(structure.node_structure_space[n.label].parents)

                for e in n.iter_outgoing_links():
                    target = e.get_target_node()
                    tlabel = b"DATA_NODE" if target.is_data_node() else target.label
                    if (e.label, tlabel) not in structure.node_structure_space[n.label].children:
                        continue
                    cindex = structure.node_structure_space[n.label].children[(e.label, tlabel)] + children_offset
                    local_structure.add(cindex)

                substructure[n.label].add(frozenset(local_structure))

        features = DenseTensorFunc.from_array([0, 0])
        for lbl, space in structure.node_structure_space.items():
            self.tensors[lbl] = {}
            ndim = (2 ** len(space.children), features.size()[0])
            real_ndim = [2] * len(space.children)
            real_ndim.append(ndim[-1])

            gold_structs = substructure[lbl]
            for parent, parent_idx in space.parents.items():
                tensor = DenseTensorFunc.zeros(ndim)

                for count, current_val_index, current_structure in iter_values(
                        len(space.children), len(space.parents)):
                    best_n_seen = max((len(gold_struct.intersection(current_structure)) + (parent_idx in gold_struct))
                                      for gold_struct in gold_structs)
                    n_unseen = len(current_structure) + 1 - best_n_seen

                    features[0] = max(best_n_seen, 0.01)
                    features[1] = max(n_unseen, 0.01)
                    tensor[count, :] = features
                self.tensors[lbl][parent_idx] = tensor.view_shape(real_ndim)

            tensor = DenseTensorFunc.zeros(ndim)
            for count, current_val_index, current_structure in iter_values(len(space.children), len(space.parents)):
                best_n_seen = max(len(gold_struct.intersection(current_structure)) for gold_struct in gold_structs)
                n_unseen = len(current_structure) - best_n_seen

                features[0] = max(best_n_seen, 0.01)
                features[1] = max(n_unseen, 0.01)
                tensor[count, :] = features
            self.tensors[lbl][None] = tensor.view_shape(real_ndim)

    @staticmethod
    def get_instance(train_sms: List[SemanticModel]):
        if CoherenceTensors.instance is None:
            CoherenceTensors.instance = CoherenceTensors(LocalStructure.get_instance(train_sms), train_sms)

        return CoherenceTensors.instance

