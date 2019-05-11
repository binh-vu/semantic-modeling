#!/usr/bin/python
# -*- coding: utf-8 -*-
import random
from collections import OrderedDict
from itertools import islice
from numbers import Number
from typing import List, Union, Optional, Dict


class Schema(object):
    """A schema is a json object that define how data is organize in form:
        <schema> := {<prop> => <value>|<schema>, _type: list|object}
        <value> := <single_value>|<list_value>
        <prop> := <string>
        <single_value> := <token>|<number>
        <list_value> := list<token>|list<number>
        """
    SINGLE_VALUE = "single_value"
    NULL_VALUE = "null_value"
    LIST_VALUE = "list_value"
    EMPTY_LIST_VALUE = "empty_list_value"

    PATH_DELIMITER = "|"

    def __init__(self):
        self.attributes: Dict[str, Union[str, Schema]] = OrderedDict()
        self.is_list_of_objects: bool = False

    @staticmethod
    def extract(rows: List[dict]):
        """Auto extract schema from list of rows

        Algorithm:
            1. Extract schema of first row
            2. Update schema as we go

        NOTE: Number of objects can be dynamic, but number of properties must not be dynamic!, therefore, we don't have a list of a list!
        """
        assert len(rows) > 0, "Cannot figure out schema if we don't have any data"
        schema = _extract_schema(rows[0])
        schema.is_list_of_objects = True
        for row in islice(rows, 1, None):
            schema.merge_(_extract_schema(row))
        return schema

    def merge_(self, another: "Schema") -> None:
        for prop, prop_type in another.attributes.items():
            if prop not in self.attributes:
                self.attributes[prop] = prop_type
            else:
                if self.attributes[prop] == prop_type:
                    continue
                else:
                    prop_types = {self.attributes[prop], prop_type}
                    if prop_types == {Schema.SINGLE_VALUE, Schema.LIST_VALUE}:
                        self.attributes[prop] = Schema.LIST_VALUE
                    elif prop_types == {Schema.SINGLE_VALUE, Schema.NULL_VALUE}:
                        self.attributes[prop] = Schema.SINGLE_VALUE
                    elif prop_types == {Schema.EMPTY_LIST_VALUE, Schema.SINGLE_VALUE}:
                        self.attributes[prop] = Schema.SINGLE_VALUE
                    elif prop_types == {Schema.EMPTY_LIST_VALUE, Schema.LIST_VALUE}:
                        self.attributes[prop] = Schema.LIST_VALUE
                    elif isinstance(prop_type, Schema) and isinstance(self.attributes[prop], Schema):
                        self.attributes[prop].merge_(prop_type)
                        self.attributes[prop].is_list_of_objects = self.attributes[prop].is_list_of_objects or prop_type.is_list_of_objects
                    elif isinstance(self.attributes[prop], Schema) and prop_type == Schema.EMPTY_LIST_VALUE:
                        self.attributes[prop].is_list_of_objects = True
                    elif Schema.NULL_VALUE in prop_types and isinstance(self.attributes[prop], Schema):
                        pass
                    elif Schema.NULL_VALUE in prop_types and isinstance(prop_type, Schema):
                        self.attributes[prop] = prop_type
                    else:
                        raise AssertionError(
                            f"Type for prop {prop} is incompatible. Get: {self.attributes[prop]} vs {prop_type}")

    def normalize(self, row: Optional[dict]):
        """Normalize the data so it will follow its schema

        a property has value None indicate missing values, however, there are some exceptions:
        + a list of objects will be empty list instead of none
        + a dict will be a dict will all property is None (not empty dict)

        the information isn't same, however, in the scope of this project, we ONLY want to figure out the
        configuration (semantic mapping); and therefore, the nuance of missing values won't affect to our
        decision (only the people who write data transformer need to care); and simply the definition make our
        job much easier.
        """
        object = {}
        if row is None:
            for attr, val in self.attributes.items():
                if isinstance(val, Schema):
                    if val.is_list_of_objects:
                        object[attr] = []
                    else:
                        object[attr] = val.normalize(None)
                else:
                    object[attr] = None
            return object

        for attr, val in self.attributes.items():
            # if it is none, then its value is missing
            row_val: Optional[Union[dict, list, str, Number]] = row.get(attr, None)
            if isinstance(val, Schema):
                if val.is_list_of_objects:
                    if row_val is None:
                        row_val = []
                    if not isinstance(row_val, list):
                        row_val = [row_val]

                    object[attr] = [val.normalize(r) for r in row_val]
                else:
                    object[attr] = val.normalize(row_val)
            elif val == Schema.LIST_VALUE:
                if row_val is not None and not isinstance(row_val, list):
                    row_val = [row_val]
                object[attr] = row_val
            else:
                object[attr] = row_val
        return object

    def get_attr_paths(self) -> List[str]:
        """Return list of path of every attributes in the schema (including nested attributes)"""
        paths = []
        for attr, val in self.attributes.items():
            if isinstance(val, Schema):
                for apath in val.get_attr_paths():
                    paths.append(attr + Schema.PATH_DELIMITER + apath)
            else:
                paths.append(attr)
        return paths

    def has_attr_path(self, attr_path: str) -> bool:
        schema = self
        attrs = attr_path.split(Schema.PATH_DELIMITER)

        for attr in islice(attrs, 0, len(attrs) - 1):
            if attr not in schema.attributes:
                return False
            schema = schema.attributes[attr]

        return attrs[-1] in schema.attributes

    def add_new_attr_path(self, attr_path: str, attr_type: str, after_attr_path: Optional[str]=None) -> None:
        schema = self
        attrs = attr_path.split(Schema.PATH_DELIMITER)

        for attr in islice(attrs, 0, len(attrs) - 1):
            schema = schema.attributes[attr]

        if after_attr_path is not None:
            after_attrs = after_attr_path.split(Schema.PATH_DELIMITER)
            need_move_to_lasts = []
            start_moving = False
            for key in schema.attributes:
                if start_moving:
                    need_move_to_lasts.append(key)
                if key == after_attrs[-1]:
                    start_moving = True
            schema.attributes[attrs[-1]] = attr_type
            for key in need_move_to_lasts:
                schema.attributes.move_to_end(key, True)
        else:
            schema.attributes[attrs[-1]] = attr_type

    def delete_attr_path(self, attr_path: str) -> None:
        idx = attr_path.find(Schema.PATH_DELIMITER)
        if idx == -1:
            self.attributes.pop(attr_path)
            return

        attr = attr_path[:idx]
        self.attributes[attr].delete_attr_path(attr_path[idx + 1:])
        if len(self.attributes[attr].attributes) == 0:
            self.attributes.pop(attr)

    def update_attr_path(self, attr_path: str, attr_type: str) -> None:
        idx = attr_path.find(Schema.PATH_DELIMITER)
        if idx == -1:
            self.attributes[attr_path] = attr_type
            return

        attr = attr_path[:idx]
        self.attributes[attr].update_attr_path(attr_path[idx + 1:], attr_type)

    def get_attr_type(self, attr_path: str) -> str:
        attrs = attr_path.split(Schema.PATH_DELIMITER)
        schema = self
        for attr in islice(attrs, 0, len(attrs) - 1):
            schema = schema.attributes[attr]
        return schema.attributes[attrs[-1]]

    def get_nested_schema(self, path: List[str]) -> 'Schema':
        schema = self
        for attr in path:
            schema = schema.attributes[attr]
        return schema

    def clone(self):
        return Schema.from_dict(self.to_dict())

    def to_dict(self):
        return {
            "attributes": [(attr, val if not isinstance(val, Schema) else val.to_dict()) for attr, val in self.attributes.items()],
            "is_list_of_objects": self.is_list_of_objects
        }

    @staticmethod
    def from_dict(obj: dict):
        schema = Schema()
        schema.is_list_of_objects = obj['is_list_of_objects']
        for attr, val in obj['attributes']:
            if isinstance(val, dict):
                schema.attributes[attr] = Schema.from_dict(val)
            else:
                schema.attributes[attr] = val

        return schema


def _extract_schema(obj: dict) -> Schema:
    attributes = OrderedDict()
    for prop, val in obj.items():
        if isinstance(val, list):
            if len(val) == 0:
                # want to preserve information that this is an empty list, later, it can be a list of objects, or list of values
                attributes[prop] = Schema.EMPTY_LIST_VALUE
            else:
                assert not isinstance(val[0], list), "#properties cannot be dynamic"
                if isinstance(val[0], dict):
                    attributes[prop] = _extract_schema(val[0])
                    for v in islice(val, 1, None):
                        attributes[prop].merge_(_extract_schema(v))
                    attributes[prop].is_list_of_objects = True
                else:
                    attributes[prop] = Schema.LIST_VALUE
        elif isinstance(val, dict):
            attributes[prop] = _extract_schema(val)
        else:
            if val is None:
                attributes[prop] = Schema.NULL_VALUE
            else:
                attributes[prop] = Schema.SINGLE_VALUE

    schema = Schema()
    schema.attributes = attributes
    return schema
