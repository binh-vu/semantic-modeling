#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from pathlib import Path
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from semantic_modeling.config import config
from semantic_modeling.data_io import get_data_tables, get_raw_data_tables, get_semantic_models, get_ontology
from semantic_modeling.utilities.serializable import serializeJSON
from transformation.r2rml.commands.modeling import SetInternalLinkCmd, SetSemanticTypeCmd
from transformation.r2rml.r2rml import R2RML

dataset = "museum_edm"
ont = get_ontology(dataset)

r2rml_dir = Path(config.datasets[dataset].as_path()) / "karma-version" / "models-r2rml"
r2rml_dir.mkdir(exist_ok=True, parents=True)
model_dir = Path(config.datasets[dataset].models_y2rml.as_path())

for tbl in get_data_tables(dataset):
    r2rml_file = r2rml_dir / f"{tbl.id}-model.ttl"
    r2rml = R2RML.load_from_file(model_dir / f"{tbl.id}-model.yml")
    # note that we use a cleaned data table, whatever columns need to create/transform have been done.
    # therefore, we will remove all command that aren't SetSemanticType or SetInternalLink
    r2rml.commands = [cmd for cmd in r2rml.commands if isinstance(cmd, (SetSemanticTypeCmd, SetInternalLinkCmd))]
    r2rml.to_kr2rml(ont, tbl, r2rml_file)