#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, Tuple, List, Set, Union, Optional

from semantic_modeling.config import config
from semantic_modeling.data_io import get_ontology
from transformation.models.data_table import DataTable
from transformation.service.kr2rml import KR2RML


"""add yaml files so it's easier to read"""

dataset = "museum_edm"
ont = get_ontology(dataset)
workdir = Path(config.datasets[dataset].as_path())
(workdir / "models-y2rml").mkdir(exist_ok=True, parents=True)

for file in sorted((workdir / "sources").iterdir()):
    # if not file.name.startswith("s04"):
    #     continue
    r2rml_file = workdir / "models-r2rml" / f"{file.stem}-model.ttl"
    tbl = DataTable.load_from_file(file)
    print(tbl.head(10).to_string("double"))
    kr2rml = KR2RML(ont, tbl, r2rml_file)
    # tbl = kr2rml.apply_build(tbl)[0]
    # kr2rml.to_yaml(workdir / "models-y2rml" / f"{file.stem}-model.yml")
    # print(tbl.head(10).to_string("double"))
