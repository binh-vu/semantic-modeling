#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import pickle

import ujson, yaml
from collections import OrderedDict
from pathlib import Path
from typing import Union, Optional


def serialize(obj, fpath: Union[Path, str]):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def serializeCSV(array, fpath: Union[Path, str], delimiter=","):
    with open(fpath, "w") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, delimiter=delimiter)
        for line in array:
            writer.writerow(line)


def serializeJSON(obj, fpath: Union[Path, str], indent: int=0):
    with open(fpath, 'w') as f:
        # add a special handler to handle case of list of instances of some class
        if type(obj) is list:
            if len(obj) > 0 and hasattr(obj[0], 'to_dict'):
                return ujson.dump((o.to_dict() for o in obj), f, indent=indent)
        elif hasattr(obj, 'to_dict'):
            return ujson.dump(obj.to_dict(), f, indent=indent)

        ujson.dump(obj, f, indent=indent)


def deserialize(fpath: Union[Path, str]):
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def deserializeJSON(fpath: Union[Path, str], Class=None):
    with open(fpath, 'r') as f:
        obj = ujson.load(f)
        if Class is not None:
            if type(obj) is list:
                return [Class.from_dict(o) for o in obj]
            else:
                return Class.from_dict(obj)

        return obj


def deserializeCSV(fpath: Union[Path, str], quotechar: str='"'):
    with open(fpath, "r") as f:
        reader = csv.reader(f, quotechar=quotechar)
        return [row for row in reader]


def serialize2str(obj) -> bytes:
    return pickle.dumps(obj)


def deserialize4str(bstr: bytes):
    return pickle.loads(bstr)

# noinspection PyPep8Naming
def deserializeYAML(fpath: Union[Path, str]) -> dict:
    # load yaml with OrderedDict to preserve order
    # http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    def load_yaml_file(file_stream):
        # noinspection PyPep8Naming
        def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
            class OrderedLoader(Loader):
                pass

            # noinspection PyArgumentList
            def construct_mapping(loader, node):
                loader.flatten_mapping(node)
                return object_pairs_hook(loader.construct_pairs(node))

            # noinspection PyUnresolvedReferences
            OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
            return yaml.load(stream, OrderedLoader)

        # noinspection PyTypeChecker
        return ordered_load(file_stream, yaml.SafeLoader)

    with open(fpath, 'r') as f:
        return load_yaml_file(f)


# noinspection PyPep8Naming
def serializeYAML(obj: dict, fpath: Union[Path, str]) -> None:
    def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwargs):
        class OrderedDumper(Dumper):
            pass

        def _dict_representer(dumper, data):
            return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwargs)

    with open(fpath, 'w') as f:
        ordered_dump(obj, f, default_flow_style=False, indent=4)