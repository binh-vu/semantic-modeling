#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import TYPE_CHECKING, List

from terminaltables import DoubleTable, AsciiTable, build
from terminaltables.width_and_alignment import max_dimensions

from semantic_modeling.algorithm.string import auto_wrap
from transformation.models.table_schema import Schema

if TYPE_CHECKING:
    from transformation.models.data_table import DataTable


def visualize(tbl: 'DataTable', format: str="ascii", inner_row_border: bool=True, max_text_width: int=40):
    if format == 'double':
        clazz = DoubleTable
    elif format == 'ascii':
        clazz = AsciiTable
    else:
        assert False

    def render_tbl(title, arrays, meta_dim):
        inst = clazz(arrays)
        inst.title = title
        inst.inner_heading_row_border = False
        inst.inner_row_border = inner_row_border
        dimensions = max_dimensions(inst.table_data, inst.padding_left, inst.padding_right)[:3]
        return build.flatten(inst.gen_table(meta_dim['current_dim'][0], dimensions[1], meta_dim['current_dim'][1]))

    def wrap_text(schema: Schema, row):
        object = {}
        for attr, val in schema.attributes.items():
            if isinstance(val, Schema):
                if val.is_list_of_objects:
                    object[attr] = [wrap_text(val, r) for r in row[attr]]
                else:
                    object[attr] = wrap_text(val, row[attr])
            elif val == Schema.LIST_VALUE:
                if row[attr] is None:
                    object[attr] = None
                else:
                    object[attr] = [auto_wrap(s, max_text_width) if isinstance(s, str) else s for s in row[attr]]
            else:
                if isinstance(row[attr], str):
                    object[attr] = auto_wrap(row[attr], max_text_width)
                else:
                    object[attr] = row[attr]
        return object

    def cal_dimensions(schema: Schema, rows: List[dict]):
        """Calculate dimensions of a table so that we can render it properly"""
        meta = {
            "current_dim": None,
            "nested_dims": {}
        }
        placeholder = {}
        for attr, val in schema.attributes.items():
            if isinstance(val, Schema):
                if val.is_list_of_objects:
                    meta['nested_dims'][attr], sample_str = cal_dimensions(val, [r for row in rows for r in row[attr]])
                else:
                    meta['nested_dims'][attr], sample_str = cal_dimensions(val, [row[attr] for row in rows])
                placeholder[attr] = sample_str
            elif val == Schema.LIST_VALUE:
                nested_arrays = [[r] for row in rows if row[attr] is not None for r in row[attr]]
                instance = clazz(nested_arrays[:1])
                dimensions = max_dimensions(nested_arrays, instance.padding_left, instance.padding_right)[:3]
                meta['nested_dims'][attr] = {
                    "current_dim": (dimensions[0], dimensions[2]),
                    "nested_dims": {}
                }
                placeholder[attr] = build.flatten(instance.gen_table(dimensions[0], dimensions[1][:1], dimensions[2]))

        arrays = [[]]
        for attr, val in schema.attributes.items():
            if isinstance(val, Schema):
                arrays[-1].append(placeholder[attr])
            else:
                arrays[-1].append(attr)

        for row in rows:
            array = []
            for attr, val in schema.attributes.items():
                if isinstance(val, Schema) or val == Schema.LIST_VALUE:
                    array.append(placeholder[attr])
                else:
                    array.append(row[attr])
            arrays.append(array)

        instance = clazz([arrays[0]])
        dimensions = max_dimensions(arrays, instance.padding_left, instance.padding_right)[:3]
        meta['current_dim'] = dimensions[0], dimensions[2]

        sample_str = build.flatten(instance.gen_table(dimensions[0], dimensions[1][:1], dimensions[2]))
        return meta, sample_str

    def schema2array(schema: Schema, meta_dims: dict):
        array = []
        for attr, val in schema.attributes.items():
            if isinstance(val, Schema):
                tbl_str = render_tbl(attr, [schema2array(val, meta_dims['nested_dims'][attr])], meta_dims['nested_dims'][attr])
                array.append(tbl_str)
            else:
                array.append(attr)
        return array

    def row2array(schema: Schema, row: dict, meta_dims: dict):
        array = []
        for attr, val in schema.attributes.items():
            if isinstance(val, Schema):
                dim = meta_dims['nested_dims'][attr]
                if val.is_list_of_objects:
                    tbl_str = render_tbl(None, [row2array(val, r, dim) for r in row[attr]], dim)
                    array.append(tbl_str)
                else:
                    tbl_str = render_tbl(None, [row2array(val, row[attr], dim)], dim)
                    array.append(tbl_str)
            elif val == Schema.LIST_VALUE:
                dim = meta_dims['nested_dims'][attr]
                if row[attr] is not None:
                    tbl_str = render_tbl(None, [[str(x)] for x in row[attr]], dim)
                    array.append(tbl_str)
                else:
                    array.append(row[attr])
            else:
                array.append(row[attr])

        return array

    rows = [wrap_text(tbl.schema, r) for r in tbl.rows]
    meta_dims = cal_dimensions(tbl.schema, rows)[0]

    arrays = []
    arrays.append(schema2array(tbl.schema, meta_dims))
    for row in rows:
        arrays.append(row2array(tbl.schema, row, meta_dims))

    return render_tbl(tbl.id, arrays, meta_dims)
    # x = clazz(arrays)
    # x.title = tbl.title
    # x.inner_row_border = inner_row_border
    # x.inner_heading_row_border = False
    # return str(x.table)
