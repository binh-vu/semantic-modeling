#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional


_counter = 0


def next_number() -> int:
    global _counter
    _counter += 1
    return _counter


def reset_number():
    global _counter
    _counter = 0
