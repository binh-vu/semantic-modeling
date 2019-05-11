#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from typing import Dict, Tuple, List, Union, Optional, Callable

from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

import numpy
from numpy.core.multiarray import dot
from numpy.linalg import norm

from semantic_labeling.column import Column


def jaccard_sim_test(col1: Column, col2: Column):
    col1data = set(col1.get_textual_data())
    col2data = set(col2.get_textual_data())

    if len(col1data) == 0 or len(col2data) == 0:
        return 0

    return len(col1data.intersection(col2data)) / len(col1data.union(col2data))


def cosine_similarity(vec1: numpy.ndarray, vec2: numpy.ndarray) -> float:
    norm1 = norm(vec1)
    norm2 = norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot(vec1, vec2) / (norm1 * norm2)


def get_tokenizer() -> Tokenizer:
    infixes = [
        '(?<=[0-9A-Za-z])[\\.](?=[0-9])',
        '(?<=[0-9])[\\.](?=[0-9A-Za-z])',
    ]
    English.Defaults.infixes = tuple(list(English.Defaults.infixes) + infixes)
    return English.Defaults.create_tokenizer()
