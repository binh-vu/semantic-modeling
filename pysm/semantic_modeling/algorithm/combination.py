#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any


def iter_index(array_lengths: List[int]):
    """Iterate all combination of value in list of arrays"""
    current_val_index = [0] * len(array_lengths)
    yield current_val_index

    while True:
        # iterate through each assignment
        i = len(current_val_index) - 1
        while i >= 0:
            # move to next state & set value of variables to next state value
            current_val_index[i] += 1
            if current_val_index[i] == array_lengths[i]:
                current_val_index[i] = 0
                i = i - 1
            else:
                break

        if i < 0:
            # already iterated through all values
            break

        # yield current values
        yield current_val_index
