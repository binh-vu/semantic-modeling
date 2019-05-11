#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Dict, List, Union, Optional

from data_structure import GraphNode
from semantic_modeling.karma.semantic_model import SemanticModel
from transformation.models.data_table import DataTable
from transformation.models.scope import Scope
from transformation.models.table_schema import Schema


class ExtractorConfiguration:

    def __init__(self) -> None:
        super().__init__()
        self.attributes = {}
        self.scope: Scope = Scope("")

    def add_attribute(self, lbl: str, attribute: Union['AttrPath', 'ExtractorConfiguration']):
        if lbl not in self.attributes:
            self.attributes[lbl] = [attribute]
        else:
            self.attributes[lbl].append(attribute)

    def to_dict(self):
        return {
            "scope": str(self.scope),
            "attributes": {
                k: [v.to_dict() if isinstance(v, ExtractorConfiguration) else str(v) for v in vs] for k, vs in self.attributes.items()
            }
        }

class AttrPath:

    def __init__(self, path: str, is_relative: bool):
        self.path = path.split(Schema.PATH_DELIMITER)
        self.is_relative = is_relative

    def get_value(self, scoped_row: dict, original_row: dict):
        if self.is_relative:
            if len(self.path) == 1 and self.path[0] == "":
                return scoped_row
            return self._val(scoped_row, self.path)
        return self._val(original_row, self.path)

    def _val(self, row: dict, attr_path: List[str]):
        if len(attr_path) == 1:
            return row[attr_path[0]]

        if isinstance(row[attr_path[0]], list):
            return [self._val(v, attr_path[1:]) for v in row[attr_path[0]]]

        return self._val(row[attr_path[0]], attr_path[1:])

    def __str__(self):
        if self.is_relative:
            return f"::{Schema.PATH_DELIMITER.join(self.path)}"
        return Schema.PATH_DELIMITER.join(self.path)


class Navigator(object):
    """To navigate in/out between scope

    {
        people: [
            { name: jay }
            { name: minh }
        ],
        partner: [{
            name: IBM,
            ceo: {
                name: BlaBla,
                phone: 213-213-213
            }
        }],
        department: {
            name: Viterbi,
            address: {
                city: LA,
                country: US
            }
        }  # nav: department
    }

    some scopes:
        people.1
        department
        department.address
        partner.0.ceo
    """
    def __init__(self):
        self.current_paths: List[Union[str, int]] = []
        self.current_path_objs: list = []

    def push(self, attr: Union[str, int], val):
        self.current_paths.append(attr)
        self.current_path_objs.append(val)
        return self

    def pop(self):
        return self.current_paths.pop(), self.current_path_objs.pop()

    def replace(self, attr: Union[str, int], val):
        self.current_paths[-1] = attr
        self.current_path_objs[-1] = val

    def remove(self):
        self.current_paths.pop()
        self.current_path_objs.pop()
        return self

    def is_empty(self):
        return len(self.current_paths) == 0

    def __repr__(self):
        return Schema.PATH_DELIMITER.join((str(x) for x in self.current_paths))


def sm2extractor(schema: Schema, attr2type: Dict[str, str], node: GraphNode):
    """Create an extractor from a semantic mapping. An extractor is a nested dictionary
    that encode an attribute path to a data node
    """
    extractor = ExtractorConfiguration()
    extractor.attributes['@type'] = node.label.decode('utf-8')

    dnodes = []
    id = None

    for link in node.iter_outgoing_links():
        target = link.get_target_node()
        if target.is_data_node():
            dnodes.append(target)
        if link.label == b"karma:classLink":
            id = link

    dnode2lbl = {dnode.id: dnode.label.decode("utf-8") for dnode in dnodes}

    # create default scope, if we have id (karma:classLink), then the scope is the id scope, otherwise, the scope has to
    # follow the default rule: one object per row!!
    if id is not None:
        if len(dnodes) == 1 and attr2type[dnode2lbl[id.target_id]] == Schema.LIST_VALUE:
            # one exception, list of objects
            scope = Scope(dnode2lbl[id.target_id])
        else:
            scope = Scope(dnode2lbl[id.target_id]).get_parent()
        extractor.scope = scope
    else:
        if len(dnodes) == 1 and attr2type[dnode2lbl[dnodes[0].id]] == Schema.LIST_VALUE:
            # one exception, list of objects
            extractor.scope = Scope(dnode2lbl[dnodes[0].id])
        elif len(dnodes) > 0:
            # expect all properties belong to same nested object; otherwise they have to add id column
            _scopes = [Scope(dnode2lbl[dnode.id]).get_parent() for dnode in dnodes]
            _scope  = min(_scopes)
            _nested_schema = schema.get_nested_schema(_scope.attr_paths)

            for s in _scopes:
                if s != _scope:
                    # make sure there are no path from _scope to s go through a list
                    tschema = _nested_schema
                    for attr in s.attr_paths[len(_scope.attr_paths):]:
                        tschema = tschema.attributes[attr]
                        assert not tschema.is_list_of_objects

            extractor.scope = _scope

    for link in node.iter_outgoing_links():
        target = link.get_target_node()
        if target.is_class_node():
            extractor.add_attribute(link.label.decode("utf-8"), sm2extractor(schema, attr2type, target))

    for dnode in dnodes:
        dlink = dnode.get_first_incoming_link()
        if extractor.scope.contain_path(dnode2lbl[dnode.id]):
            attr_path = AttrPath(extractor.scope.get_relative_path(dnode2lbl[dnode.id]), is_relative=True)
            extractor.add_attribute(dlink.label.decode('utf-8'), attr_path)
        else:
            extractor.add_attribute(dlink.label.decode("utf-8"), AttrPath(dnode.label.decode("utf-8"), False))

    return extractor


def extract_inner(nav: Navigator, extractor: ExtractorConfiguration, relative_paths: List[str], local_row: dict, global_row: dict):
    attr = relative_paths[0]

    if len(relative_paths) == 1:
        if isinstance(local_row[attr], list):
            results = []
            nav = nav.push(attr, local_row[attr]).push(0, None)
            for i, val in enumerate(local_row[attr]):
                nav.replace(i, val)
                results.append(extract(nav, extractor, val, global_row))
            nav.remove().remove()
            return results
        else:
            nav.push(attr, local_row[attr])
            result = extract(nav, extractor, local_row[attr], global_row)
            nav.remove()
            return result

    if isinstance(local_row[attr], list):
        results = []
        nav.push(attr, local_row[attr]).push(0, None)
        for i, val in enumerate(local_row[attr]):
            nav.replace(i, val)
            results.append(extract_inner(nav, extractor, relative_paths[1:], val, global_row))
        nav.remove().remove()
        return results

    nav.push(attr, local_row[attr])
    result = extract_inner(nav, extractor, relative_paths[1:], local_row[attr], global_row)
    nav.remove()
    return result


def extract_outer(navigator: Navigator, extractor: ExtractorConfiguration, global_row: dict):
    scope_idx = 0
    relative_paths = []
    for i, attr in enumerate(extractor.scope.attr_paths):
        if isinstance(navigator.current_paths[scope_idx], int):
            scope_idx += 1
        else:
            if navigator.current_paths[scope_idx] != attr:
                relative_paths = extractor.scope.attr_paths[i:]
                break
        scope_idx += 1

    # need to capture current path so we can restore later
    path_stack = []
    for i in range(len(navigator.current_paths) - scope_idx):
        path_stack.append(navigator.pop())

    if navigator.is_empty():
        if len(relative_paths) == 0:
            val = extract(navigator, extractor, global_row, global_row)
        else:
            val = extract_inner(navigator, extractor, relative_paths, global_row, global_row)
    else:
        val = extract_inner(navigator, extractor, relative_paths, navigator.current_path_objs[-1], global_row)

    for arg in reversed(path_stack):
        navigator.push(*arg)
    return val


def extract(navigator: Navigator, extractor: ExtractorConfiguration, local_row: dict, global_row: dict) -> Optional[dict]:
    if local_row is None:
        return None

    object = {}
    for attr, confs in extractor.attributes.items():
        if attr[0] == '@':
            object[attr] = confs
            continue

        for conf in confs:
            if isinstance(conf, ExtractorConfiguration):
                # this is another class node
                if extractor.scope.is_same_scope(conf.scope):
                    # same scope, don't need to navigate
                    val = extract(navigator, conf, local_row, global_row)
                elif extractor.scope.is_outer_scope_of(conf.scope):
                    # navigate to the inner scope, it may be a list of objects, or an individual object
                    val = extract_inner(navigator, conf, extractor.scope.get_relative_path2scope(conf.scope), local_row, global_row)
                else:
                    # need to navigate to outer scope, it cannot be a list
                    val = extract_outer(navigator, conf, global_row)
            else:
                val = conf.get_value(local_row, global_row)

            if attr in object:
                if val is None:
                    # we only skip None, when we know this is a list of objects intead a single value
                    continue

                if not isinstance(object[attr], list):
                    object[attr] = [object[attr]]
                object[attr].append(val)
            else:
                object[attr] = val

    if all(object[attr] is None for attr in extractor.attributes if not attr[0].startswith('@')):
        return None

    return object


def jsonld_generator(sm: SemanticModel, tbl: DataTable):
    root = [n for n in sm.graph.iter_class_nodes() if n.n_incoming_links == 0][0]
    attr2type = {apath: tbl.schema.get_attr_type(apath) for apath in tbl.schema.get_attr_paths()}
    extractor = sm2extractor(tbl.schema, attr2type, root)

    results = []
    for row in tbl.rows:
        nav = Navigator()
        if extractor.scope.path == "":
            if isinstance(row, list):
                results += [extract(nav, extractor, v, v) for v in row]
            else:
                results.append(extract(nav, extractor, row, row))
        else:
            res = extract_inner(nav, extractor, extractor.scope.attr_paths, row, row)
            if isinstance(res, list):
                results += res
            else:
                results.append(res)

    return results
