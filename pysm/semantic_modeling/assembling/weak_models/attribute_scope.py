#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

from semantic_modeling.data_io import get_semantic_models
from transformation.models.table_schema import Schema


class AttributeScope:

    instance = None

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.attribute_same_scope_matrix: Dict[str, Dict[Tuple[bytes, bytes], bool]] = {}
        for sm in get_semantic_models(dataset):
            self.attribute_same_scope_matrix[sm.id] = {}
            attr_paths = [attr.label.split(Schema.PATH_DELIMITER) for attr in sm.attrs]
            for i in range(len(sm.attrs)):
                for j in range(i + 1, len(sm.attrs)):
                    is_same_scope = attr_paths[i][:-1] == attr_paths[j][:-1]
                    self.attribute_same_scope_matrix[sm.id][(sm.attrs[i].label.encode('utf-8'), sm.attrs[j].label.encode('utf-8'))] = is_same_scope
                    self.attribute_same_scope_matrix[sm.id][(sm.attrs[j].label.encode('utf-8'), sm.attrs[i].label.encode('utf-8'))] = is_same_scope

    @staticmethod
    def get_instance(dataset: str):
        if AttributeScope.instance is None:
            AttributeScope.instance = AttributeScope(dataset)

        assert AttributeScope.instance.dataset == dataset
        return AttributeScope.instance

    def is_same_scope(self, sm_id: str, attr_x: bytes, attr_y: bytes) -> bool:
        return self.attribute_same_scope_matrix[sm_id][(attr_x, attr_y)]