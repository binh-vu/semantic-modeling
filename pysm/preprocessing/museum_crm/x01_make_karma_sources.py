#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson
from pathlib import Path
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from semantic_modeling.config import config
from semantic_modeling.data_io import get_data_tables, get_raw_data_tables, get_semantic_models, get_ontology, \
    get_sampled_data_tables
from semantic_modeling.utilities.serializable import serializeJSON
from transformation.r2rml.commands.modeling import SetInternalLinkCmd, SetSemanticTypeCmd
from transformation.r2rml.r2rml import R2RML

dataset = "museum_crm"
ont = get_ontology(dataset)

source_dir = Path(config.datasets[dataset].as_path()) / "karma-version" / "sources"
source_dir.mkdir(exist_ok=True, parents=True)
for tbl in get_sampled_data_tables(dataset):
    serializeJSON(tbl.rows, source_dir / f"{tbl.id}.json", indent=4)