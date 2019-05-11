#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional

from transformation.models.table_schema import Schema


class Scope:

    def __init__(self, path: str):
        self.path = path
        if path == "":
            self.attr_paths = []
        else:
            self.attr_paths = path.split(Schema.PATH_DELIMITER)

    def is_outer_scope_of(self, scope: 'Scope') -> bool:
        return scope.path.startswith(self.path) and scope.path != self.path

    def is_same_scope(self, scope: 'Scope') -> bool:
        return scope.path == self.path

    def get_parent(self):
        return Scope(Schema.PATH_DELIMITER.join(self.attr_paths[:-1]))

    def get_inner_scope(self):
        assert len(self.attr_paths) > 0
        return Scope(Schema.PATH_DELIMITER.join(self.attr_paths[1:]))

    def contain_path(self, path: str):
        return path.startswith(self.path)

    def get_relative_path(self, path: str):
        if self.path == "":
            return path
        return path[len(self.path)+1:]

    def get_relative_path2scope(self, scope: 'Scope'):
        """Return a relative path to another scope"""
        return scope.attr_paths[len(self.attr_paths):]

    def extract_data(self, global_row: dict):
        if self.path == "":
            return global_row
        return _extract_data(self.attr_paths, global_row)

    def __eq__(self, other):
        if other is None or not isinstance(other, Scope):
            return False

        return self.path == other.path

    def __lt__(self, other):
        if other is None or not isinstance(other, Scope):
            raise NotImplementedError()

        return other.path.startswith(self.path) and other.path != self.path

    def __gt__(self, other):
        if other is None or not isinstance(other, Scope):
            raise NotImplementedError()

        return self.path.startswith(other.path) and other.path != self.path

    def __repr__(self):
        return self.path


def _extract_data(attr_paths: List[str], local_row: dict):
    attr = attr_paths[0]
    if len(attr_paths) == 1:
        return local_row[attr]

    for attr in attr_paths:
        if isinstance(local_row[attr], list):
            return [_extract_data(attr_paths[1:], val) for val in local_row[attr]]
        return _extract_data(attr_paths[1:], local_row[attr])
