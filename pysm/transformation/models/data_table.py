#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from copy import copy, deepcopy
from itertools import islice
from pathlib import Path
from typing import List

import numpy

from transformation.models.scope import Scope
from transformation.models.table_schema import Schema
from transformation.utils.data_reader import load_xml, load_json, load_csv
from transformation.utils.visualization import visualize


class DataTable(object):
    """Represent a `class` of data sources that we can transform using semantic mapping.

    A transformation process takes a list of rows, each row corresponds to an top-level entity, using a semantic mapping
    to map the data in each row to a graph representation (domain ontology)
    """
    VIZ_ASCII = "ascii"
    VIZ_DOUBLE = "double"

    def __init__(self, id: str, schema: Schema, rows: List[dict]):
        self.schema = schema
        self.id = id
        self.rows = rows

    @staticmethod
    def load_from_file(fpath: Path, n_rows: int=999999999):
        if fpath.suffix == ".xml":
            rows = load_xml(fpath)
        elif fpath.suffix == ".json" or fpath.suffix == ".jl":
            rows = load_json(fpath)
        elif fpath.suffix == ".csv":
            # first row always headers!!
            rows = load_csv(fpath, header=True)
        else:
            assert False, f"Invalid file: {fpath}"

        # we need to have rows in a form of a list of rows, otherwise, we cannot distinguish between whether a property
        #  of an entity is a list or a single value
        if isinstance(rows, dict):
            while isinstance(rows, dict):
                if len(rows) > 1:
                    raise AssertionError(
                        "Trying to un-roll a dict to a list or rows, but its have more than one property: %s" % list(
                            rows.keys()))

                rows = next(iter(rows.values()))

        # filter to make rows unique
        unique_rows_index = set()
        unique_rows = []
        for row in islice(rows, n_rows):
            row_key = ujson.dumps(row)  # not always provide consistent result, but we hope...
            if row_key not in unique_rows_index:
                unique_rows_index.add(row_key)
                unique_rows.append(row)

        tbl = DataTable.load_from_rows(fpath.stem, unique_rows)
        return tbl

    @staticmethod
    def load_from_rows(title: str, rows: List[dict]):
        schema = Schema.extract(rows)
        tbl = DataTable(title, schema, [schema.normalize(r) for r in rows])
        return tbl

    def rebuild_schema(self):
        self.schema = Schema.extract(self.rows)

    def head(self, n: int):
        return DataTable(self.id, self.schema, self.rows[:n])

    def clone(self):
        return DataTable(copy(self.id), self.schema.clone(), deepcopy(self.rows))

    def sample(self, n: int, seed: int):
        if len(self.rows) <= n:
            return self

        random_state = numpy.random.RandomState(seed)
        sample_index = random_state.choice(numpy.arange(len(self.rows)), size=n, replace=False)

        rows = [self.rows[idx] for idx in sample_index]
        return DataTable(self.id, self.schema, rows)

    def get_data_in_scope(self, scope: Scope):
        return [scope.extract_data(row) for row in self.rows]

    def to_string(self, *args):
        return visualize(self, *args)

    def to_dict(self) -> dict:
        return {
            "title": self.id,
            "schema": self.schema.to_dict(),
            "rows": self.rows,
        }

    @staticmethod
    def from_dict(obj: dict) -> "DataTable":
        return DataTable(obj['title'], Schema.from_dict(obj['schema']), obj['rows'])