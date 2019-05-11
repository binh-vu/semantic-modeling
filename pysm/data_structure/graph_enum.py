#!/usr/bin/python
# -*- coding: utf-8 -*-
from enum import IntEnum


class GraphLinkType(IntEnum):
    UNSPECIFIED = 0
    UNKNOWN = 1
    URI_PROPERTY = 2
    OBJECT_PROPERTY = 3
    DATA_PROPERTY = 4


class GraphNodeType(IntEnum):
    CLASS_NODE = 1
    DATA_NODE = 2
