#!/usr/bin/python
# -*- coding: utf-8 -*-
from functools import wraps
from typing import Dict, Tuple, List, Union, Optional, Callable, Any

import inspect
import redis
import ujson

from semantic_modeling.config import config
from semantic_modeling.utilities.serializable import serialize2str, deserialize4str


class RedisConnection(object):

    instance: 'RedisConnection' = None

    def __init__(self) -> None:
        self.redis = redis.StrictRedis(host=config.redis.host, port=config.redis.port, db=config.redis.db)

    @staticmethod
    def get_instance():
        if RedisConnection.instance is None:
            RedisConnection.instance = RedisConnection()
        return RedisConnection.instance

    def set_data(self, key: str, value: Any) -> None:
        self.redis.set(key, value)
        # redis auto back up should be enable. e.g: save 60 1
        # self.redis.save()

    def get_data(self, key: str) -> Any:
        return self.redis.get(key)

    def has_data(self, key: str) -> bool:
        return self.redis.exists(key)


def redis_cache_func(get_arg_id: Dict[str, Callable[[Any], str]], func_path: str):
    def wrapper(func):
        _args = func.__code__.co_varnames[:func.__code__.co_argcount]
        _kwargs = {kw: i for i, kw in enumerate(_args)}

        @wraps(func)
        def cached_func(*args, **kwargs):
            key_arg = {kw: None for kw in _args}
            for i, arg in enumerate(args):
                if _args[i] in get_arg_id:
                    arg = get_arg_id[_args[i]](arg)

                key_arg[_args[i]] = arg
            for kw, arg in kwargs.items():
                if kw in get_arg_id:
                    arg = get_arg_id[kw](arg)
                key_arg[kw] = arg

            for kw, arg in key_arg.items():
                assert isinstance(arg, (int, str, bool)), 'Cannot cache function has argument type: `%s`' % (type(arg),)

            key_arg = '%s:%s(%s)' % (func_path, func.__name__, ujson.dumps(key_arg))
            redis = RedisConnection.get_instance()
            if not redis.has_data(key_arg):
                result = func(*args, **kwargs)
                redis.set_data(key_arg, serialize2str(result))
            else:
                result = redis.get_data(key_arg)
                result = deserialize4str(result)

            return result
        return cached_func
    return wrapper