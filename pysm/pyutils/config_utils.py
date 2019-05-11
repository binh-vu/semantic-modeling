#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
import shutil
from collections import OrderedDict
from typing import Dict, Iterable, TypeVar, Union

import yaml


class RemoteOSPath(object):

    # remote path: scheme://[host]:[port]/ or it could be scheme://user@host:port/
    remote_path_reg = re.compile(r'''([a-zA-Z]+://[a-zA-Z0-9:]*)(/.*)?''')

    @staticmethod
    def join(parent_path, child_path):
        # if the child path is a fully remote path, then it is remote abs path
        if RemoteOSPath.remote_path_reg.match(child_path):
            return child_path

        match = RemoteOSPath.remote_path_reg.match(parent_path)
        if match is None:
            return os.path.join(parent_path, child_path)

        remote_host, remote_path = match.groups()
        return remote_host + RemoteOSPath.abspath(os.path.join(remote_path, child_path))

    @staticmethod
    def abspath(path):
        match = RemoteOSPath.remote_path_reg.match(path)
        if match is not None:
            remote_host, remote_path = match.groups()
            return remote_host + os.path.abspath(remote_path)
        return os.path.abspath(path)


class StringConf(str):
    # noinspection PyInitNewSignature,PyTypeChecker
    def __new__(cls, string: str, workdir: str) -> 'StringConf':
        # customize the constructor if needed
        obj = super(StringConf, cls).__new__(cls, string)
        obj.__workdir = workdir
        return obj

    def __add__(self, s: str) -> str:
        return StringConf(super().__add__(s), self.__workdir)

    def as_int(self) -> int:
        return int(self)

    def as_float(self) -> float:
        return float(self)

    def as_path(self) -> 'StringConf':
        return StringConf(RemoteOSPath.abspath(RemoteOSPath.join(self.__workdir, self)), self.__workdir)

    def ensure_path_existence(self) -> None:
        """Ensure the path existed
            1. If path is pointing to a file (using extension test), then make sure its directory existed
            2. If path is pointing to a dir (otherwise), then make sure it existed
        """
        path = self.as_path()
        _, ext = os.path.splitext(path)

        if ext != '':
            # this is a file
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        else:
            if not os.path.exists(path):
                os.makedirs(path)

    def backup_path(self) -> None:
        """Back up the path
            1. if the path doesn't exist: ensure it existed
            2. otherwise, create backup files        
        """
        path = self.as_path()
        dirname = os.path.dirname(path)
        basename = os.path.basename(path)

        if os.path.exists(path):
            # find the most recent back up
            backup_reg = re.compile('^-\d+$')
            backup_versions = [
                int(fname.replace(basename + '-', '')) for fname in os.listdir(dirname)
                if fname.startswith(basename) and backup_reg.match(fname.replace(basename, '')) is not None
            ]
            if len(backup_versions) == 0:
                most_recent_version = 0
            else:
                most_recent_version = max(backup_versions)

            # do the backup
            shutil.move(path, os.path.join(dirname, basename + '-' + str(most_recent_version + 1)))

        # create new folder to work with
        self.ensure_path_existence()


RawPrimitiveType = TypeVar('RawPrimitiveType', int, float, str)
PrimitiveType = TypeVar('PrimitiveType', int, float, StringConf)


class ListConf(object):
    def __init__(self, array: list, workdir: str) -> None:
        self.array = array
        self.workdir = workdir

    def __getitem__(self, item):
        return self.array[item]

    def __setitem__(self, item, value):
        self.array[item] = value

    def __delitem__(self, item):
        del self.array[item]

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return len(self.array)

    def as_path(self) -> str:
        return RemoteOSPath.abspath(self.workdir)

    def to_list(self):
        return self.array

    def to_raw_list(self):
        list_object = []
        for v in self.array:
            if isinstance(v, Configuration):
                v = v.to_dict()
            elif isinstance(v, ListConf):
                v = v.to_raw_list()
            elif isinstance(v, StringConf):
                v = str(v)
            list_object.append(v)
        return list_object


class Configuration(object):
    def __init__(self, dict_object: Dict[str, Union[RawPrimitiveType, Dict]], workdir: str='', init: bool=True) -> None:
        # if __workdir__ is defined in dict_object, it overwrites the bounded workdir
        if '__workdir__' in dict_object:
            workdir = RemoteOSPath.join(workdir, dict_object['__workdir__'])

        self.__dict_object: Dict = dict_object
        self.__workdir: str = workdir
        self.__conf: OrderedDict[str, Union[PrimitiveType, Configuration]] = OrderedDict()

        for key, value in dict_object.items():
            if key in {'__workdir__'}:
                continue

            if isinstance(value, (dict, OrderedDict)):
                self.__conf[key] = Configuration(value, workdir, False)
            elif isinstance(value, str):
                self.__conf[key] = StringConf(value, workdir)
            elif type(value) is list and len(value) > 0:
                if isinstance(value[0], str):
                    self.__conf[key] = ListConf([StringConf(x, workdir) for x in value], workdir)
                elif isinstance(value[0], dict):
                    self.__conf[key] = ListConf([Configuration(x, workdir, False) for x in value], workdir)
                else:
                    self.__conf[key] = value
            else:
                self.__conf[key] = value

        if init:
            self.defer_init(self, self)

    def defer_init(self, global_conf: 'Configuration', config: Union[ListConf, 'Configuration']) -> None:
        """Initialize value in config"""
        if isinstance(config, ListConf):
            for i, item in enumerate(config):
                if isinstance(item, StringConf):
                    if item.startswith('@@'):
                        # value is a reference to other value as path
                        item = global_conf.get_conf(item[2:]).as_path()
                    elif item.startswith('@#'):
                        # value is interpret as path
                        item = StringConf(item[2:], config.workdir).as_path()
                    elif item.startswith('@'):
                        item = global_conf.get_conf(item[1:])
                    config[i] = item
                elif isinstance(item, ListConf):
                    self.defer_init(global_conf, item)
                elif isinstance(item, Configuration):
                    self.defer_init(global_conf, item)
        else:
            for prop in list(config.__conf.keys()):
                value = config.__conf[prop]
                if isinstance(value, StringConf):
                    if value.startswith('@@'):
                        # value is a reference to other value as path
                        value = global_conf.get_conf(value[2:]).as_path()
                    elif value.startswith('@#'):
                        # value is interpret as path
                        value = StringConf(value[2:], config.__workdir).as_path()
                    elif value.startswith('@'):
                        # value is a reference to other value
                        value = global_conf.get_conf(value[1:])
                    config.__conf[prop] = value
                elif isinstance(value, ListConf):
                    self.defer_init(global_conf, value)
                elif isinstance(value, Configuration):
                    self.defer_init(global_conf, value)

    def set_conf(self, key: str, value: RawPrimitiveType, split_key: bool=True) -> 'Configuration':
        if type(value) is dict:
            value = Configuration(value, self.__workdir)
        elif type(value) is str:
            value = StringConf(value, self.__workdir)
        elif type(value) is list and len(value) > 0:
            if isinstance(value[0], str):
                value = ListConf([StringConf(x, self.__workdir) for x in value], self.__workdir)
            elif isinstance(value[0], dict):
                value = ListConf([Configuration(x, self.__workdir, False) for x in value], self.__workdir)
                self.defer_init(self, value)

        conf = self
        if split_key:
            p_keys = key.split('.')
        else:
            p_keys = [key]

        for p_key in p_keys[:-1]:
            assert type(conf) is Configuration, 'Cannot assign property to primitive object'
            if p_key not in conf.__conf:
                conf.__conf[p_key] = Configuration(OrderedDict(), self.__workdir)

            conf = conf.__conf[p_key]

        assert type(conf) is Configuration, 'Cannot assign property to primitive object'
        conf.__conf[p_keys[-1]] = value

        return self

    def get_conf(self, key: str) -> Union[PrimitiveType, 'Configuration']:
        """Get configuration provided by the dot string"""
        conf = self
        props = key.split('.')
        for prop in props[:-1]:
            conf = conf.__conf[prop]
        return conf.__conf[props[-1]]

    def __getattr__(self, name: str) -> Union[PrimitiveType, 'Configuration']:
        return self.__conf[name]

    def __getitem__(self, name: str) -> Union[PrimitiveType, 'Configuration']:
        return self.__conf[name]

    def __iter__(self) -> Iterable[str]:
        return iter(self.__conf.keys())

    def __contains__(self, item: str) -> bool:
        return item in self.__conf

    def __delitem__(self, item):
        self.__conf.pop(item)

    def __delattr__(self, item):
        self.__conf.pop(item)

    def as_path(self) -> str:
        return RemoteOSPath.abspath(self.__workdir)

    def items(self):
        return self.__conf.items()

    def to_dict(self, including_workdir=False) -> Dict[str, Union[RawPrimitiveType, Dict]]:
        dict_object = {}
        if including_workdir:
            dict_object['__workdir__'] = self.__workdir

        for k, v in self.__conf.items():
            if isinstance(v, Configuration):
                v = v.to_dict()
            elif isinstance(v, ListConf):
                v = v.to_raw_list()
            elif isinstance(v, StringConf):
                v = str(v)
            dict_object[k] = v
        return dict_object

    def to_raw_dict(self):
        return self.__dict_object


def load_config(fpath: str) -> Configuration:
    # load yaml with OrderedDict to preserve order
    # http://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    def load_yaml_file(file_stream):
        def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
            class OrderedLoader(Loader):
                pass

            def construct_mapping(loader, node):
                loader.flatten_mapping(node)
                return object_pairs_hook(loader.construct_pairs(node))

            OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
            return yaml.load(stream, OrderedLoader)

        # noinspection PyTypeChecker
        return ordered_load(file_stream, yaml.SafeLoader)

    with open(fpath, 'r') as f:
        return Configuration(load_yaml_file(f), workdir=os.path.dirname(fpath))


def write_config(config: Configuration, fpath: str) -> None:
    def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwargs):
        class OrderedDumper(Dumper):
            pass

        def _dict_representer(dumper, data):
            return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwargs)

    with open(fpath, 'w') as f:
        ordered_dump(config.to_raw_dict(), f, default_flow_style=False, indent=4)
