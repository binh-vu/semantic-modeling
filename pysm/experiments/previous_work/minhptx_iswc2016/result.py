#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Contain interface to communicate with semantic labeling API"""
from pathlib import Path
from typing import Dict, List


class MinhptxSemanticType(object):

    def __init__(self, weight: float, domain: str, type: str) -> None:
        self.weight = weight
        self.domain = domain
        self.type = type


class MinhptxSemanticLabelingResult(object):
    """Store source labeling result"""

    def __init__(self) -> None:
        super().__init__()
        self.columns: Dict[str, List[MinhptxSemanticType]] = {}

    def set_column(self, col_name: str, col_types: List[MinhptxSemanticType]) -> None:
        self.columns[col_name] = col_types

    @staticmethod
    def from_dict(obj: dict) -> Dict[str, 'MinhptxSemanticLabelingResult']:
        res = {}
        for source_id, source_labeling in obj.items():
            source_id = Path(source_id).stem
            res[source_id] = MinhptxSemanticLabelingResult()
            for col_name, col_types in source_labeling.items():
                col_stypes = []
                for weight, types in col_types:
                    for type in types:
                        domain, type = type.split("---")
                        col_stypes.append(MinhptxSemanticType(weight, domain, type))

                res[source_id].set_column(col_name, col_stypes)
        return res
