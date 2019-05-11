#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from pathlib import Path
from typing import List, Union

from semantic_modeling.settings import Settings
from semantic_modeling.config import config, get_logger
from semantic_modeling.karma.karma import KarmaModel
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserialize, deserializeJSON, serialize, serializeJSON
from transformation.models.data_table import DataTable
from transformation.r2rml.r2rml import R2RML

_data_io_vars = {
    "ont": {},
    "karma_models": {},
    "semantic_models": {},
    "data_tables": {},
    "raw_data_tables": {},
    "sampled_data_tables": {}
}
_logger = get_logger("app.data_io")


def get_ontology(dataset: str) -> Ontology:
    """Get ontology of a given dataset"""
    global _data_io_vars
    if dataset not in _data_io_vars["ont"]:
        # if it has been cached ...
        cache_file = get_cache_dir(dataset) / 'ont.pkl'
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        if cache_file.exists():
            ont = deserialize(cache_file)
        else:
            ont = Ontology.from_dataset(dataset)
            serialize(ont, cache_file)
        _data_io_vars["ont"][dataset] = ont

    return _data_io_vars["ont"][dataset]


def get_short_train_name(train_sms: Union[List[str], List[SemanticModel]]):
    if isinstance(train_sms[0], SemanticModel):
        names = sorted([sm.id[:3] for sm in train_sms])
    else:
        names = sorted([sm[:3] for sm in train_sms])

    assert len(names) == len(set(names)), "prefix of semantic models must be unique"
    if all(name[0] == "s" and name[1:].isdigit() for name in names):
        groups = [[names[0]]]
        for i, name in enumerate(names[1:], 1):
            if int(names[i - 1][1:]) + 1 == int(name[1:]):
                # consequetive
                groups[-1].append(name)
            else:
                groups.append([name])

        name = "--".join([
            g[0] if len(g) == 1 else f"{g[0]}-to-{g[-1]}"
            for g in groups
        ])
    else:
        name = "-".join(names)
    return name


def get_cache_dir(dataset: str, train_sms: Union[List[str], List[SemanticModel]]=None) -> Path:
    if train_sms is None:
        cached_dir = Path(config.fsys.debug.as_path()) / dataset / "cached"
    else:
        name = get_short_train_name(train_sms)
        cached_dir = Path(config.fsys.debug.as_path()) / dataset / "cached" / f"train-sm-{name}"

    cached_dir.mkdir(exist_ok=True, parents=True)
    return cached_dir


def get_work_dir(dataset: str, train_sms: Union[List[str], List[SemanticModel]]=None) -> Path:
    if train_sms is None:
        workdir = Path(config.fsys.debug.as_path()) / dataset / "workdir"
    else:
        name = get_short_train_name(train_sms)
        workdir = Path(config.fsys.debug.as_path()) / dataset / "workdir" / f"train-sm-{name}"

    workdir.mkdir(exist_ok=True, parents=True)
    return workdir


def get_karma_models(dataset: str) -> List[KarmaModel]:
    """Get list of json models of a given dataset"""
    global _data_io_vars

    if dataset not in _data_io_vars["karma_models"]:
        # if it has been cached...
        cache_file = get_cache_dir(dataset) / 'karma_models.json'
        if cache_file.exists():
            karma_models = deserializeJSON(cache_file, Class=KarmaModel)
        else:
            karma_models = []
            model_dir = Path(config.datasets[dataset].karma_version.as_path()) / "models-json"
            ont = get_ontology(dataset)
            for file in sorted(model_dir.iterdir()):
                if file.name.endswith(".json"):
                    karma_models.append(KarmaModel.load_from_file(ont, file))
            serializeJSON(karma_models, cache_file)
        _data_io_vars["karma_models"][dataset] = karma_models

    return _data_io_vars["karma_models"][dataset]


def get_raw_data_tables(dataset: str) -> List[DataTable]:
    global _data_io_vars
    if dataset not in _data_io_vars['raw_data_tables']:
        # if it has been cached...
        cache_file = get_cache_dir(dataset) / 'raw_tables.pkl'
        if cache_file.exists():
            raw_tables = deserialize(cache_file)
        else:
            raw_tables = []
            source_dir = Path(config.datasets[dataset].data.as_path())
            for file in sorted(source_dir.iterdir()):
                if file.name.startswith("."):
                    continue
                raw_tables.append(DataTable.load_from_file(file))

            serialize(raw_tables, cache_file)
        _data_io_vars["raw_data_tables"][dataset] = raw_tables

    return _data_io_vars["raw_data_tables"][dataset]


def get_data_tables(dataset: str) -> List[DataTable]:
    global _data_io_vars
    if dataset not in _data_io_vars['data_tables']:
        # if it has been cached...
        cache_file = get_cache_dir(dataset) / 'tables.pkl'
        if cache_file.exists():
            tables = deserialize(cache_file)
        else:
            mapping_dir = Path(config.datasets[dataset].models_y2rml.as_path())
            raw_tables = get_raw_data_tables(dataset)
            R2RML.load_python_scripts(Path(config.datasets[dataset].python_code.as_path()))
            tables = []
            semantic_models = []
            for i, raw_tbl in enumerate(raw_tables):
                r2rml_file = mapping_dir / f"{raw_tbl.id}-model.yml"
                tbl, sm = R2RML.load_from_file(r2rml_file).apply_build(raw_tbl)
                tables.append(tbl)
                semantic_models.append(sm)

            serialize(tables, cache_file)
            _data_io_vars['semantic_models'][dataset] = semantic_models  # avoid apply R2RML twice!

        _data_io_vars["data_tables"][dataset] = tables

    return _data_io_vars["data_tables"][dataset]


def get_sampled_data_tables(dataset: str) -> List[DataTable]:
    global _data_io_vars
    if dataset not in _data_io_vars['sampled_data_tables']:
        # if it has been cached...
        cache_file = get_cache_dir(dataset) / "sampled_tables.pkl"
        if cache_file.exists():
            tables = deserialize(cache_file)
        else:
            tables = get_data_tables(dataset)
            settings = Settings.get_instance()
            tables = [tbl.sample(settings.n_samples, settings.random_seed) for tbl in tables]
            serialize(tables, cache_file)
        _data_io_vars["sampled_data_tables"][dataset] = tables

    return _data_io_vars["sampled_data_tables"][dataset]


def clear_sampled_data_tables(dataset: str) -> None:
    cache_file = get_cache_dir(dataset) / "sampled_tables.pkl"
    if cache_file.exists():
        os.remove(str(cache_file))


def get_semantic_models(dataset: str) -> List[SemanticModel]:
    """Get list of semantic models of a given dataset"""
    global _data_io_vars

    if dataset not in _data_io_vars["semantic_models"]:
        # if it has been cached...
        cache_file = get_cache_dir(dataset) / 'semantic_models.json'
        if cache_file.exists():
            semantic_models = deserializeJSON(cache_file, Class=SemanticModel)
        else:
            mapping_dir = Path(config.datasets[dataset].models_y2rml.as_path())
            R2RML.load_python_scripts(Path(config.datasets[dataset].python_code.as_path()))
            raw_tables = get_raw_data_tables(dataset)
            semantic_models = []
            tables = []
            for i, raw_tbl in enumerate(raw_tables):
                r2rml_file = mapping_dir / f"{raw_tbl.id}-model.yml"
                tbl, sm = R2RML.load_from_file(r2rml_file).apply_build(raw_tbl)
                semantic_models.append(sm)
                tables.append(tbl)

            serializeJSON(semantic_models, cache_file)
            _data_io_vars["data_tables"][dataset] = tables

        _data_io_vars["semantic_models"][dataset] = semantic_models

    return _data_io_vars["semantic_models"][dataset]


if __name__ == '__main__':
    dataset = 'museum_crm'
    ont = Ontology.from_dataset(dataset)

    data_dir = Path(config.datasets[dataset].as_path())
    (data_dir / "models-viz").mkdir(exist_ok=True, parents=True)
    (data_dir / "tables-viz").mkdir(exist_ok=True, parents=True)

    for sm in get_semantic_models(dataset):
        sm.graph.render2pdf(data_dir / f"models-viz/{sm.id}.pdf")

    for tbl in get_data_tables(dataset):
        with open(data_dir / "tables-viz" / f"{tbl.id}.txt", "wb") as f:
            f.write(tbl.to_string().encode("utf-8"))
