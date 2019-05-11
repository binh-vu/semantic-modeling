#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import ujson
from pathlib import Path

from experiments.arg_helper import str2bool
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_ontology, \
    get_sampled_data_tables, clear_sampled_data_tables
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.serializable import serializeJSON, deserializeJSON


def get_shell_args():
    parser = argparse.ArgumentParser('Assembling experiment')
    parser.register("type", "boolean", str2bool)

    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--n_samples', type=int, required=True, default=1000, help='Number of samples')
    parser.add_argument('--seed', type=int, required=True, default=120, help='Random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_shell_args()
    dataset = args.dataset

    settings = Settings.get_instance(False)
    settings.n_samples = args.n_samples
    settings.random_seed = args.seed
    settings.log_current_settings()

    ont = get_ontology(dataset)
    source_dir = Path(config.datasets[dataset].as_path()) / "karma-version" / "sources"
    source_dir.mkdir(exist_ok=True, parents=True)
    meta_file = source_dir / ".meta"

    if meta_file.exists():
        meta = deserializeJSON(meta_file)

        if meta['n_samples'] == settings.n_samples and meta['random_seed'] == settings.random_seed:
            print(
                "Don't need to prepare karma sources because it has been generated with same configuration before. Terminating...!")
            exit(0)

    print(f"Generate karma sources for dataset: {dataset}")
    serializeJSON({
        'n_samples': settings.n_samples,
        'random_seed': settings.random_seed
    }, meta_file, indent=4)

    model_dir = Path(config.datasets[dataset].models_y2rml.as_path())
    # clear cache file
    clear_sampled_data_tables(dataset)

    for tbl, sm in zip(get_sampled_data_tables(dataset), get_semantic_models(dataset)):
        serializeJSON(tbl.rows, source_dir / f"{tbl.id}.json", indent=4)
