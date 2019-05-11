#!/usr/bin/python
# -*- coding: utf-8 -*-
import enum
from ctypes import c_char_p, c_float
from multiprocessing import Array
from typing import Dict, List, Union, Optional


class ColumnType(enum.Enum):

    NUMBER = "number"
    STRING = "string"
    DATETIME = "datetime"
    NULL = "null"

    def is_comparable(self):
        return self == ColumnType.NUMBER or self == ColumnType.DATETIME


class Column(object):

    def __init__(
        self, table_name: str, name: str, type: ColumnType, size: int, type_stats: Dict[ColumnType, float]
    ):
        self.id = f"{table_name}:{name}"
        self.table_name = table_name
        self.name = name
        self.size = size
        self.type_stats = type_stats
        self.type = type
        self.value: Optional[ColumnData] = None

    def get_id(self) -> str:
        return self.id

    def get_n_null(self) -> int:
        return round(self.type_stats[ColumnType.NULL] * self.size)

    def get_numeric_data(self) -> list:
        return self.value.number_arrays

    def get_numeric_data_index(self) -> list:
        return self.value.number_idx_arrays

    def get_textual_data(self) -> list:
        return self.value.string_arrays

    def get_textual_data_index(self) -> list:
        return self.value.string_idx_arrays

    def to_dict(self) -> dict:
        return {
            "table_name": self.table_name,
            "name": self.name,
            "type": self.type.value,
            "size": self.size,
            "type_stats": {t.value: v
                           for t, v in self.type_stats.items()}
        }

    @staticmethod
    def from_dict(val) -> 'Column':
        return Column(val["table_name"], val["name"], ColumnType(val["type"]), val["size"],
            {ColumnType(t): v
             for t, v in val["type_stats"].items()}
        )


class SharedColumnData(object):

    def __init__(self, arrays: List[Union[int, float, bytes, None]]) -> None:
        self.number_arrays = None
        self.string_arrays = None

        number_arrays = []
        string_arrays = []
        for val in arrays:
            if val is not None:
                if isinstance(val, (int, float)):
                    number_arrays.append(val)
                else:
                    string_arrays.append(val)

        self.number_arrays = Array(c_float, number_arrays, lock=False)
        self.string_arrays = Array(c_char_p, len(string_arrays), lock=False)
        for i, val in enumerate(string_arrays):
            self.string_arrays[i] = val


class ColumnData(object):

    def __init__(self, arrays: List[Union[int, float, bytes, None]]) -> None:
        self.number_arrays = []
        self.number_idx_arrays = []
        self.string_arrays = []
        self.string_idx_arrays = []

        for i, val in enumerate(arrays):
            if val is not None:
                if isinstance(val, (int, float)):
                    self.number_arrays.append(val)
                    self.number_idx_arrays.append(i)
                else:
                    self.string_arrays.append(val)
                    self.string_idx_arrays.append(i)
