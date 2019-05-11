#!/usr/bin/python
# -*- coding: utf-8 -*-

import fastnumbers
import ujson
from typing import Dict, Union, List

from semantic_labeling.column import Column, ColumnType, ColumnData
from semantic_modeling.config import get_logger
from transformation.models.data_table import DataTable
from transformation.models.table_schema import Schema


def guess_type(val, empty_as_null: bool) -> ColumnType:
    """Guess type of a value"""
    if val is None:
        return ColumnType.NULL

    assert isinstance(val, (int, float, str)), "Invalid column data"
    if fastnumbers.isfloat(val):
        return ColumnType.NUMBER
    else:
        if len(val.strip()) == 0 and empty_as_null:
            return ColumnType.NULL
        return ColumnType.STRING


def get_type(val) -> ColumnType:
    if val is None:
        return ColumnType.NULL

    if type(val) is float:
        return ColumnType.NUMBER
    else:
        return ColumnType.STRING


class ListValueException(Exception):
    pass


def norm_val(val, empty_as_null: bool) -> Union[bytes, int, float, None]:
    """Normalize a value"""
    if val is None:
        return None

    if fastnumbers.isfloat(val) or fastnumbers.isint(val):
        return fastnumbers.float(val)

    val = val.strip()
    if len(val) == 0 and empty_as_null:
        return None

    return val.encode("utf-8", "ignore")


def get_col_values(attr_paths: List[str], row: dict, col_values: list):
    """Get value of a columns, flatten any nested list"""
    attr = attr_paths[0]
    if len(attr_paths) == 1:
        if isinstance(row[attr], list):
            col_values += row[attr]
        else:
            col_values.append(row[attr])
        return

    if isinstance(row[attr], list):
        for val in row[attr]:
            get_col_values(attr_paths[1:], val, col_values)
    else:
        get_col_values(attr_paths[1:], row[attr], col_values)


class ColumnBasedTable(object):

    logger = get_logger('app.semantic_labeling.data_table')

    def __init__(self, id: str, columns: List[Column]) -> None:
        self.id = id
        self.columns: List[Column] = columns
        self.name2colidx: Dict[str, int] = {cname.name: idx for idx, cname in enumerate(columns)}

    def get_column_by_name(self, name: str):
        return self.columns[self.name2colidx[name]]

    @staticmethod
    def from_table(tbl: DataTable) -> 'ColumnBasedTable':
        columns = []
        for cname in tbl.schema.get_attr_paths():
            type_stats = {type: 0.0 for type in [ColumnType.NUMBER, ColumnType.STRING, ColumnType.NULL]}
            col_values = []
            for row in tbl.rows:
                get_col_values(cname.split(Schema.PATH_DELIMITER), row, col_values)

            col_values = [norm_val(val, empty_as_null=True) for val in col_values]
            for val in col_values:
                type_stats[get_type(val)] += 1

            for key, val in type_stats.items():
                type_stats[key] = val / len(col_values)

            # now we have to decide what type of this column using some heuristic!!
            if type_stats[ColumnType.STRING] > type_stats[ColumnType.NUMBER]:
                col_type = ColumnType.STRING
            else:
                if type_stats[ColumnType.NULL] < 0.7 and (type_stats[ColumnType.NUMBER] + type_stats[ColumnType.NULL]) < 0.9:
                    col_type = ColumnType.STRING
                elif type_stats[ColumnType.NUMBER] > type_stats[ColumnType.STRING] and (type_stats[ColumnType.NUMBER] + type_stats[ColumnType.NULL]) > 0.9:
                    col_type = ColumnType.NUMBER
                else:
                    if all(val is None for val in col_values):
                        col_type = ColumnType.NULL
                    else:
                        ColumnBasedTable.logger.error("Cannot decide type with the stats: %s", ujson.dumps(type_stats, indent=4))
                        raise Exception(f"Cannot decide type of column: {col_name} in {tbl.id}")
            column = Column(tbl.id, cname, col_type, len(col_values), type_stats)
            column.value = ColumnData(col_values)
            columns.append(column)

        col_based_tbl = ColumnBasedTable(tbl.id, columns)
        return col_based_tbl

    def to_dict(self):
        return {
            "id": self.id,
            "columns": [col.to_dict() for col in self.columns]
        }

    @staticmethod
    def from_dict(val) -> 'ColumnBasedTable':
        tbl = ColumnBasedTable(val["id"], [
            Column.from_dict(col) for col in val["columns"]
        ])
        return tbl

    # implement pickling
    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        obj = ColumnBasedTable.from_dict(state)
        self.__dict__ = obj.__dict__
