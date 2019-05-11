#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Optional
import json_schema
import os

import ujson

from semantic_modeling.config import config

# %%
# ################ Print schema of models-json
fpaths: List[str] = [
    os.path.join(config.datasets.museum_edm.models_json.as_path(), fname)
    for fname in os.listdir(config.datasets.museum_edm.models_json.as_path())
]
models = []

for fpath in fpaths:
    with open(fpath, 'r') as f:
        models.append(ujson.load(f))

schema = json_schema.generate_schema(models)
print(schema.to_string(indent=4))
