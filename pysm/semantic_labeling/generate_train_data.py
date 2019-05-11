#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Optional

import numpy

from semantic_labeling.sm_type_db import SemanticTypeDB


def generate_training_data(stype_db: SemanticTypeDB) -> Tuple[list, list, list, list]:
    x_train, x_test = [], []
    y_train, y_test = [], []

    for i, ref_col in enumerate(stype_db.train_columns):
        for j, col in enumerate(stype_db.train_columns):
            if i == j:
                continue
            x_train.append(stype_db.similarity_matrix[j, i])
            y_train.append(stype_db.col2types[col.id] == stype_db.col2types[ref_col.id])

        for j, col in enumerate(stype_db.test_columns):
            x_test.append(stype_db.similarity_matrix[j, i])
            y_test.append(stype_db.col2types[col.id] == stype_db.col2types[ref_col.id])

    return x_train, y_train, x_test, y_test

