#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Contains pyutils for csv
"""
import csv
from io import StringIO
from typing import List, Any


def dump_csv(array: List[Any], delimiter=',') -> str:
    """Dump csv array to string

    Example:
        >>> dump_csv(["black", "yellow", "green"])
        'black,yellow,green'

        >>> dump_csv(["black", "yellow", "green"], delimiter='|')
        'black|yellow|green'
    """
    f = StringIO()
    writer = csv.writer(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(array)

    return f.getvalue()[:-2]
