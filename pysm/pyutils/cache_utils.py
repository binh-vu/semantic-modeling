#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import ujson
from typing import Dict, Any, Callable


class Cache(object):

    def __init__(self) -> None:
        self.data = {}

    def exec_func(self, func: Callable[[Any], Any], *args: Any) -> Any:
        key = ujson.dumps(args)
        if func.__name__ not in self.data:
            self.data[func.__name__] = {}

        if key not in self.data[func.__name__]:
            self.data[func.__name__][key] = func(*args)
        return self.data[func.__name__][key]

    def clear_func(self, func: Callable[[Any], Any]):
        del self.data[func.__name__]


class FileCache(object):

    def __init__(self, fpath: str) -> None:
        self.fpath = fpath
        self.data = {}  # type: Dict[str, Any]
        self.fcursor = None  # type: Any
        self.within_context = False

    def load_data(self) -> None:
        # Load data if the file's existed
        assert not self.within_context, 'Must load the data before caching'
        if os.path.exists(self.fpath):
            with open(self.fpath, 'r') as f:
                for l in f:
                    k, v = ujson.loads(l)
                    self.data[k] = v

    def __enter__(self) -> None:
        self.fcursor = open(self.fpath, mode='a')
        self.within_context = True

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.within_context = False
        self.fcursor.close()

    def invalidate(self) -> None:
        if self.within_context:
            assert self.fcursor is not None
            # attempt to close the file and open new one
            self.fcursor.close()
            self.fcursor = open(self.fpath, mode='w')
        else:
            # if not in the context, mean the file is not opened
            assert self.fcursor is None
        self.data = {}

    def exec_func(self, func: Callable[[Any], Any], *args: Any) -> Any:
        assert self.within_context

        key = '%s:%s' % (func.__name__, ujson.dumps(args))
        if key not in self.data:
            self.data[key] = func(*args)
            self.write_change(key)

        return self.data[key]

    def write_change(self, key: str) -> None:
        self.fcursor.write(ujson.dumps((key, self.data[key])))
        self.fcursor.write('\n')


class FileCacheDelegator(object):
    def __init__(self, fpath: str, object_constructor: Callable[[], object]) -> None:
        self.file_cache: FileCache = FileCache(fpath)
        self.object_constructor: Callable[[], object] = object_constructor
        self.object: object = None  # type: object
        self.delegator = {}  # type: Dict[str, Callable[[*Any], Any]]

    def load_data(self) -> None:
        self.file_cache.load_data()

    def __enter__(self):
        self.file_cache.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_cache.__exit__(exc_type, exc_val, exc_tb)

    def invalidate(self) -> None:
        self.file_cache.invalidate()

    def __get_delegate_func(self, func_name: str) -> Callable[[Any], Any]:
        def delegate(*args):
            if self.object is None:
                self.object = self.object_constructor()
            return getattr(self.object, func_name)(*args)
        delegate.__name__ = func_name

        return delegate

    def __get_exec_func(self, func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def exec_func(*args):
            return self.file_cache.exec_func(func, *args)

        return exec_func

    def __getattr__(self, item: str) -> Callable[[Any], Any]:
        """Alias of the cached function"""

        if item not in self.delegator:
            self.delegator[item] = self.__get_exec_func(self.__get_delegate_func(item))

        exec_func = self.delegator[item]
        return exec_func
