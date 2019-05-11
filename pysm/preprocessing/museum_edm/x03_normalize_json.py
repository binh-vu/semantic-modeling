#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from semantic_modeling.config import config
from semantic_modeling.utilities.serializable import deserializeJSON, serializeJSON

"""Usually run after generate r2rml and copied from KARMA_HOME"""

dataset = "museum_edm"
model_dir = Path(config.datasets[dataset].karma_version.as_path()) / "models-json"
for file in sorted(model_dir.iterdir()):
    sm = deserializeJSON(file)
    sm['id'] = Path(sm['id']).stem
    sm['name'] = sm['id']

    serializeJSON(sm, model_dir / f"{sm['id']}-model.json", indent=4)
    os.remove(file)