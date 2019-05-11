#!/usr/bin/python
# -*- coding: utf-8 -*-
import uuid
from numbers import Number
from typing import Dict, List, Union

import numpy
from pathlib import Path
from pyutils.list_utils import _
from terminaltables import AsciiTable

from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_sampled_data_tables, get_data_tables, get_cache_dir
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import serialize, deserialize
from transformation.jsonld_generator import jsonld_generator
# %%
from transformation.models.data_table import DataTable
from transformation.models.scope import Scope
from transformation.models.table_schema import Schema


class CardinalityMatrix:

    ONE_TO_N = "1-TO-N"
    N_TO_ONE = "N-TO-1"
    ONE_TO_ONE = "1-TO-1"
    UNCOMPARABLE = "NULL"

    def __init__(self, columns: List[str], matrix: List[List[Union[str, float]]]):
        self.matrix = matrix
        self.columns = columns
        self.col2idx = {col: i for i, col in enumerate(columns)}
        self.bcol2idx = {col.encode('utf-8'): i for i, col in enumerate(columns)}

    def assign_cardinality(self, given_col: str, col: str, cardinality: Union[str, float]):
        self.matrix[self.col2idx[given_col]][self.col2idx[col]] = cardinality

    @staticmethod
    def from_table(tbl: DataTable):
        columns = tbl.schema.get_attr_paths()
        matrix = CardinalityMatrix(columns, [
            [CardinalityMatrix.UNCOMPARABLE for j in range(len(columns))]
            for i in range(len(columns))
        ])

        groups = divide_columns_to_comparable_groups(tbl)
        gmatrix = {}
        for group in groups:
            for col, val in get_subcardinality_matrix(tbl, group).items():
                gmatrix[col] = val

                if val is not None:
                    for col2, cardinality in val.items():
                        matrix.assign_cardinality(col, col2, cardinality)

        # compare between groups, try to figure relationship between groups
        gscopes = [min(Scope(c).get_parent() for c in group) for group in groups]
        for i, group in enumerate(groups):
            for j in range(len(groups)):
                if i == j:
                    continue

                if gscopes[i].is_outer_scope_of(gscopes[j]):
                    # then it must be 1-to-n relationship since it's the reason why we divide to two separate groups
                    for col in groups[i]:
                        for col2 in groups[j]:
                            matrix.assign_cardinality(col, col2, CardinalityMatrix.ONE_TO_N)
                elif gscopes[j].is_outer_scope_of(gscopes[i]):
                    # then it must be n-to-1 relationship since it's the reason why we divide to two separate groups
                    for col in groups[i]:
                        for col2 in groups[j]:
                            matrix.assign_cardinality(col, col2, CardinalityMatrix.N_TO_ONE)
                elif gscopes[i].is_same_scope(gscopes[j]):
                    # it must be 1-to-n or n-to-1, decide based on the attr_type, it must be a column which has a list value
                    # because if it is a list of objects => different scope
                    if len(groups[i]) == 1:
                        # n-to-1 relationships
                        assert tbl.schema.get_attr_type(groups[i][0]) == Schema.LIST_VALUE
                        for col in groups[i]:
                            for col2 in groups[j]:
                                matrix.assign_cardinality(col, col2, CardinalityMatrix.N_TO_ONE)
                    else:
                        # 1-to-n relationship
                        assert len(groups[j]) == 1
                        assert tbl.schema.get_attr_type(groups[j][0]) == Schema.LIST_VALUE
                        for col in groups[i]:
                            for col2 in groups[j]:
                                matrix.assign_cardinality(col, col2, CardinalityMatrix.ONE_TO_N)
                else:
                    # we simply cannot compare between two groups, just leave as it should be
                    pass

        return matrix

    def __str__(self):
        arrays = [[""] + self.columns]
        for i, row in enumerate(self.matrix):
            array = [self.columns[i]]
            for x in row:
                if isinstance(x, float):
                    array.append("%8.5f" % x)
                else:
                    array.append("%8s" % x)
            arrays.append(array)
        return AsciiTable(arrays).table


class CardinalityFeatures:

    instance = None

    def __init__(self, cardinality_matrices: Dict[str, CardinalityMatrix]):
        self.cardinality_matrices = cardinality_matrices

    def get_cardinality(self, sm_id: str, given_col: bytes, col: bytes):
        matrix = self.cardinality_matrices[sm_id]
        val = matrix.matrix[matrix.bcol2idx[given_col]][matrix.bcol2idx[col]]
        if isinstance(val, Number):
            if val > 1.05:
                return CardinalityMatrix.ONE_TO_N
            else:
                return CardinalityMatrix.ONE_TO_ONE
        return val

    @staticmethod
    def get_instance(dataset: str):
        if CardinalityFeatures.instance is None:
            cache_file = get_cache_dir(dataset) / "weak_models" / "cardinality_features.pkl"
            cache_file.parent.mkdir(exist_ok=True, parents=True)
            if not cache_file.exists():
                tables = get_data_tables(dataset)
                matrices = {}
                for tbl in tables:
                    matrices[tbl.id] = CardinalityMatrix.from_table(tbl)

                CardinalityFeatures.instance = CardinalityFeatures(matrices)
                serialize(CardinalityFeatures.instance, cache_file)
            else:
                CardinalityFeatures.instance = deserialize(cache_file)
        return CardinalityFeatures.instance

def divide_columns_to_comparable_groups(tbl: DataTable):
    groups = {}
    _divide_columns_to_comparable_groups(tbl.schema, "", groups, str(uuid.uuid4()), [])
    return list(groups.values())


def _divide_columns_to_comparable_groups(schema: Schema, parent_path: str, groups: Dict, group_idx, group):
    # return groups of columns that can compare directly with each other
    # we cannot group columns in lower scope that in a nested list (but nested object is ok)
    if parent_path != "":
        parent_path = parent_path + Schema.PATH_DELIMITER

    for attr, atype in schema.attributes.items():
        if isinstance(atype, Schema):
            if atype.is_list_of_objects:
                # create a new group and add new attributes
                _divide_columns_to_comparable_groups(atype, parent_path + attr, groups, str(uuid.uuid4()), [])
            else:
                # recursive add attribute of atype to same group
                _divide_columns_to_comparable_groups(atype, parent_path + attr, groups, group_idx, group)
        elif atype == Schema.LIST_VALUE:
            groups[str(uuid.uuid4())] = [parent_path + attr]
        else:
            group.append(parent_path + attr)

    if len(group) == 0:
        return

    if group_idx not in groups:
        groups[group_idx] = group


def _get_cardinality(rows: list, attr: str, remained_attrs: List[str]):
    grouped_rows = {}
    for row in rows:
        if row[attr] not in grouped_rows:
            grouped_rows[row[attr]] = []

        grouped_rows[row[attr]].append(row)

    average_n_values = {rattr: [] for rattr in remained_attrs}
    for attr_val, rs in grouped_rows.items():
        for rattr in remained_attrs:
            # max 2, because we are counting number of case it is one-to-N,
            # lots of time we having noises (empty, for example, is not good to use average)
            n_values = min(len({r[rattr] for r in rs}), 2)
            average_n_values[rattr].append(n_values)

    for x in average_n_values:
        average_n_values[x] = numpy.average(average_n_values[x])
    return average_n_values


def flatten_nested_dict(row: dict):
    object = {}
    for attr, val in row.items():
        if isinstance(val, dict):
            for k, v in flatten_nested_dict(val).items():
                object[attr + Schema.PATH_DELIMITER + k] = v
        else:
            object[attr] = val
    return object


def get_subcardinality_matrix(tbl: DataTable, columns: List[str]):
    scope = min(Scope(col).get_parent() for col in columns)
    relative_columns = [scope.get_relative_path(c) for c in columns]
    # columns belong to same scope, so we don't need to worry if we need information outside of scope to compare
    views = tbl.get_data_in_scope(scope)
    assert len(views) > 0
    while isinstance(views[0], list):
        views = _(views).flatten()
    views = [flatten_nested_dict(r) for r in views]

    # going to produce a matrix of comparision between columns,
    matrix = {}
    for col in columns:
        if tbl.schema.get_attr_type(col) == Schema.LIST_VALUE:
            matrix[col] = None
            continue

        remained_attrs = [c for c in columns if c != col]
        comparison = _get_cardinality(views, scope.get_relative_path(col), [scope.get_relative_path(c) for c in remained_attrs])
        matrix[col] = {}
        for x in remained_attrs:
            matrix[col][x] = comparison[scope.get_relative_path(x)]

    return matrix
