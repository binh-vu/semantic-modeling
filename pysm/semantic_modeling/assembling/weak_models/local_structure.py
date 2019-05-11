#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Dict, Tuple, List

from semantic_modeling.assembling.weak_models.multi_val_predicate import MultiValuePredicate
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel


class NodeStructureSpace:

    def __init__(self, lbl: bytes,
                 parents: Dict[Tuple[bytes, bytes], int],
                 children: Dict[Tuple[bytes, bytes], int],
                 is_class_child: List[bool],
                 child_usage_count: List[int]) -> None:
        self.lbl = lbl
        self.parents = parents
        self.children = children

        # a map from child idx => is_class_node
        self.is_class_child = is_class_child

        # a map from child_idx => count
        self.child_usage_count = child_usage_count

    def get_parent_idx(self, link_lbl: bytes, parent_lbl: bytes):
        return self.parents.get((link_lbl, parent_lbl), None)

    def get_child_idx(self, link_lbl: bytes, child_lbl: bytes):
        return self.children.get((link_lbl, child_lbl), None)


class LocalStructure(object):

    instance = None

    def __init__(self, train_sms: List[SemanticModel]) -> None:
        self.train_sm_ids = {sm.id for sm in train_sms}
        self.node_structure_space: Dict[bytes, NodeStructureSpace] = {}

        # MAKE RAW STRUCTURE SPACE FIRST
        raw_node_structure_space: Dict[bytes, dict] = {}
        for sm in train_sms:
            for n in sm.graph.iter_class_nodes():
                if n.label not in raw_node_structure_space:
                    raw_node_structure_space[n.label] = {
                        "parents": {},
                        "children": {},
                    }

                plink = n.get_first_incoming_link()
                if plink is not None:
                    triple = (plink.label, plink.get_source_node().label)
                    if triple not in raw_node_structure_space[n.label]['parents']:
                        raw_node_structure_space[n.label]['parents'][triple] = 0
                    raw_node_structure_space[n.label]['parents'][triple] += 1

                for e in n.iter_outgoing_links():
                    target = e.get_target_node()
                    triple = (e.label, b"DATA_NODE" if target.is_data_node() else target.label)
                    assert target.is_data_node()  or target.label != b"DATA_NODE"
                    if triple not in raw_node_structure_space[n.label]['children']:
                        raw_node_structure_space[n.label]['children'][triple] = 0
                    raw_node_structure_space[n.label]['children'][triple] += 1

        # MAKE NODE STRUCTURE SPACE
        for n, space in raw_node_structure_space.items():
            children_attrs = sorted(space['children'].keys(), key=lambda x: space['children'][x], reverse=True)
            self.node_structure_space[n] = NodeStructureSpace(
                n, {x: i for i, x in enumerate(space['parents'].keys())},
                {x: i for i, x in enumerate(children_attrs)},
                [x[1] != b'DATA_NODE' for x in children_attrs],
                [space['children'][x] for x in children_attrs])

    @staticmethod
    def get_instance(train_sms: List[SemanticModel]) -> 'LocalStructure':
        sm_ids = {sm.id for sm in train_sms}

        if LocalStructure.instance is None:
            LocalStructure.instance = LocalStructure(train_sms)
            return LocalStructure.instance

        assert LocalStructure.instance.train_sm_ids == sm_ids
        return LocalStructure.instance


if __name__ == '__main__':
    import ujson

    dataset = "museum_edm"
    train_size = 14
    source_models = get_semantic_models(dataset)[:train_size]

    local_structure = LocalStructure.get_instance(source_models)
    print(ujson.dumps(local_structure.node_structure_space, indent=4))