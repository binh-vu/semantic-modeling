#!/usr/bin/python
# -*- coding: utf-8 -*-

import hashlib


def md5(string: str) -> str:
    algo = hashlib.md5()
    algo.update(string)

    return algo.hexdigest()
