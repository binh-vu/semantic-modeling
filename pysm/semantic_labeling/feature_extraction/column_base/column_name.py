#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from typing import Dict, Tuple, List, Union, Optional

from semantic_labeling.column import Column
from semantic_modeling.algorithm.string import tokenize_label


def jaccard_sim_test(col1_name: str, col2_name: str, lower: bool=False) -> float:
    if lower:
        lbl1 = {x.lower() for x in tokenize_label(col1_name)}
        lbl2 = {x.lower() for x in tokenize_label(col2_name)}
    else:
        lbl1 = set(tokenize_label(col1_name))
        lbl2 = set(tokenize_label(col2_name))

    return len(lbl1.intersection(lbl2)) / len(lbl1.union(lbl2))