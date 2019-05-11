#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Optional

import shutil

import os

from experiments.previous_work.process import invoke_command
from semantic_modeling.config import config


def setup_karma(dataset: str, exec_dir: Path, json_models_dir: Optional[Path]):
    # clean all karma folder
    for fpath in exec_dir.iterdir():
        if not fpath.name.startswith("."):
            if fpath.is_dir():
                shutil.rmtree(fpath)
            else:
                os.remove(str(fpath))

    # create desired structures
    dataset_dir = Path(config.datasets[dataset].as_path())
    karma_version_dir = Path(config.datasets[dataset].karma_version.as_path())
    models_json_dir = karma_version_dir / "models-json"
    sources_dir = karma_version_dir / "sources"
    models_r2rml_dir = karma_version_dir / "models-r2rml"

    # may configure to override default folders
    if json_models_dir is not None:
        models_json_dir = json_models_dir

    shutil.copytree(models_json_dir, exec_dir / "models-json")
    shutil.copytree(sources_dir, exec_dir / "sources")
    shutil.copytree(models_r2rml_dir, exec_dir / "models-r2rml")
    shutil.copytree(dataset_dir / "ontologies", exec_dir / "preloaded-ontologies")

    for dname in ["config", "python"]:
        if (dataset_dir / dname).exists() and dname != "python":
            shutil.copytree(dataset_dir / dname, exec_dir / dname)
        else:
            os.mkdir(str(exec_dir / dname))

    for dname in ["output", "semantic-type-files", "prediction-mrr", "models-graphviz", "evaluate-mrr"]:
        (exec_dir / dname).mkdir()
    (exec_dir / "results" / "temp").mkdir(parents=True)


def execute_karma_code(
        dataset_name: str,
        exec_dir: Path,
        use_correct_type: bool,
        use_old_semantic_typer: bool,
        num_candidate_semantic_type: int,
        multiple_same_property_per_node: bool,

        w_coherence: float,
        w_confidence: float,
        w_size_reduction: float,

        num_candidate_mappings: int,
        mapping_branching_factor: int,
        topk_steiner_tree: int,
        cutoff: int,

        train_source_names: List[str],
        test_source_names: List[str]
        ):

    command = " ".join([
        "docker", "run", "--rm", "-v", f"{str(exec_dir.absolute())}:/karma-home",
        "-it", "isi/mohsen_jws2015",
        "-karma_home", "/karma-home",
        "-dataset_name", dataset_name,
        "-use_correct_type", str(use_correct_type).lower(),
        "-use_old_semantic_typer", str(use_old_semantic_typer).lower(),
        "-num_candidate_semantic_type", str(num_candidate_semantic_type).lower(),
        "-multiple_same_property_per_node", str(multiple_same_property_per_node).lower(),

        "-coefficient_coherence", str(w_coherence).lower(),
        "-coefficient_confidence", str(w_confidence).lower(),
        "-coefficient_size", str(w_size_reduction).lower(),

        "-num_candidate_mappings", str(num_candidate_mappings).lower(),
        "-mapping_branching_factor", str(mapping_branching_factor).lower(),
        "-topk_steiner_tree", str(topk_steiner_tree).lower(),
        "-cutoff", str(cutoff).lower(),

        "-train_source_names", ",".join(train_source_names),
        "-test_source_names", ",".join(test_source_names)
    ])
    invoke_command(command, exec_dir / "execution.log", output2stdout=False)


if __name__ == '__main__':
    dataset = "museum_crm"
