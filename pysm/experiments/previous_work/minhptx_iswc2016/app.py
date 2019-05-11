#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil
from itertools import chain
from pathlib import Path
from typing import Dict, List, Set, Optional, Union

import sys

from experiments.previous_work import MinhptxSemanticLabelingResult
from experiments.previous_work import invoke_command
from semantic_modeling.config import config, get_logger
from semantic_modeling.data_io import get_semantic_models, get_ontology
from semantic_modeling.karma.karma import KarmaModel
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeJSON, serializeJSON


class MinhptxSemanticLabeling(object):

    logger = get_logger("app.minhptx_iswc2016")

    def __init__(self, dataset: str, max_n_records: int=float('inf'), is_sampling: bool=False, exec_dir: Optional[Union[Path, str]]=None) -> None:
        self.dataset: str = dataset
        self.ont: Ontology = get_ontology(dataset)
        self.max_n_records: int = max_n_records
        self.is_sampling: bool = is_sampling
        assert not is_sampling, "Not implemented"

        self.source_ids: Set[str] = {
            file.stem
            for file in Path(config.datasets[dataset].data.as_path()).iterdir()
            if file.is_file() and not file.name.startswith(".")
        }

        if exec_dir is None:
            exec_dir = Path(config.fsys.debug.as_path()) / dataset / "minhptx_iswc2016"
        self.exec_dir: Path = Path(exec_dir)

        self.meta_file: Path = self.exec_dir / "execution-meta.json"
        self.input_dir: Path = self.exec_dir / "input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir: Path = self.exec_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_meta(self, train_source_ids: Set[str], test_source_ids: Set[str]):
        return {
            "dataset": self.dataset,
            "max_n_records": self.max_n_records,
            "is_sampling": self.is_sampling,
            "source_ids": self.source_ids,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "training_sources": train_source_ids,
            "testing_sources": test_source_ids
        }

    def _semantic_labeling(self, train_source_ids: Set[str],
                           test_source_ids: Set[str]) -> Dict[str, MinhptxSemanticLabelingResult]:
        """Generate semantic labeling for test_sources using train_sources"""
        need_reexec = True

        if Path(self.meta_file).exists():
            # read meta and compare if previous run is compatible with current run
            self.logger.debug("Load information from previous run...")

            meta = deserializeJSON(self.meta_file)
            meta["training_sources"] = set(meta["training_sources"])
            meta["testing_sources"] = set(meta["testing_sources"])
            meta["source_ids"] = set(meta['source_ids'])

            new_meta = self.get_meta(train_source_ids, test_source_ids)
            if len(new_meta.pop("testing_sources").difference(meta.pop("testing_sources"))) == 0:
                if new_meta == meta:
                    need_reexec = False

        if need_reexec:
            self.logger.debug("Re-execute semantic labeling...")

            try:
                # preparing data, want to compute semantic models for all sources in dataset
                data_dir = Path(config.datasets[self.dataset].data.as_path())
                model_dir = Path(config.datasets[self.dataset].models_json.as_path())

                shutil.rmtree(str(self.input_dir))
                for fpath in self.output_dir.iterdir():
                    os.remove(fpath)
                [(self.input_dir / x / y).mkdir(parents=True, exist_ok=True)
                 for x in ["%s_train" % self.dataset, "%s_test" % self.dataset]
                 for y in ["data", "model"]]

                input_train_dir = self.input_dir / ("%s_train" % self.dataset)
                input_test_dir = self.input_dir / ("%s_test" % self.dataset)

                for fpath in sorted(data_dir.iterdir()):
                    model_fname = fpath.stem + "-model.json"
                    if fpath.stem in train_source_ids:
                        self._copy_data(fpath, input_train_dir / "data" / fpath.name)
                        # seriaalize the model instead of copied because we want to convert uri to simplified uri
                        # instead of full uri (e.g karma:classLink). Full URI doesn't work in this app
                        serializeJSON(
                            KarmaModel.load_from_file(self.ont, model_dir / model_fname).to_normalized_json_model(),
                            input_train_dir / "model" / f"{fpath.name}.model.json",
                            indent=4
                        )

                    if fpath.stem in test_source_ids:
                        self._copy_data(fpath, input_test_dir / "data" / fpath.name)
                        # same reason like above
                        serializeJSON(
                            KarmaModel.load_from_file(self.ont, model_dir / model_fname).to_normalized_json_model(),
                            input_test_dir / "model" / f"{fpath.name}.model.json",
                            indent=4
                        )

                invoke_command(" ".join([
                    config.previous_works.minhptx_iswc2016.cli.as_path(), str(self.input_dir), str(self.output_dir),
                    "--train_dataset", "%s_train" % self.dataset,
                    "--test_dataset", "%s_test" % self.dataset,
                    "--evaluate_train_set", "True",
                    "--reuse_rf_model", "False"
                ]), output2file=self.exec_dir / "execution.log")
            except Exception:
                sys.stdout.flush()
                self.logger.exception("Error while preparing and invoking semantic labeling api...")
                raise

            serializeJSON(self.get_meta(train_source_ids, test_source_ids), self.meta_file, indent=4)

        # load result
        self.logger.debug("Load previous result...")
        output_files = [fpath for fpath in self.output_dir.iterdir() if fpath.suffix == ".json"]
        assert len(output_files) == 2
        app_result: Dict[str, MinhptxSemanticLabelingResult] = deserializeJSON(
            output_files[0], Class=MinhptxSemanticLabelingResult
        )
        app_result.update(deserializeJSON(output_files[1], Class=MinhptxSemanticLabelingResult))

        return {source_id: app_result[source_id] for source_id in chain(test_source_ids, train_source_ids)}

    def _copy_data(self, fsource: Path, fdest: Path) -> None:
        if self.max_n_records == float('inf'):
            shutil.copyfile(str(fsource), str(fdest))
            return

        if fsource.suffix == ".csv":
            with open(fsource, "r") as f, open(fdest, "w") as g:
                for i, line in enumerate(f):
                    if i > self.max_n_records:
                        break

                    g.write(line)
        else:
            assert False, "Not support file type: %s" % fsource.suffix

    def semantic_labeling(self, train_sources: List[SemanticModel], test_sources: List[SemanticModel], top_n: int) -> None:
        """Generate semantic labeling, and store it in test_sources"""
        train_source_ids = {s.id for s in train_sources}
        test_source_ids = {s.id for s in test_sources}
        assert len(train_source_ids.intersection(test_source_ids)) == 0
        result = self._semantic_labeling(train_source_ids, test_source_ids)

        # dump result into test_sources
        for source in chain(train_sources, test_sources):
            for col in source.attrs:
                try:
                    if col.label not in result[source.id].columns:
                        # this column is ignored
                        stypes = []
                    else:
                        stypes = result[source.id].columns[col.label]

                    col.semantic_types = [
                        KarmaSemanticType(
                            col.id, stype.domain, stype.type, "Minhptx-ISWC2016-SemanticLabeling", stype.weight
                        ) for stype in stypes
                    ][:top_n]
                except Exception:
                    self.logger.exception("Hit exception for source: %s, col: %s", source.get_id(), col.id)
                    raise


if __name__ == '__main__':
    dataset = "museum_crm"
    sources: List[SemanticModel] = get_semantic_models(dataset)[:5]

    train_size = 3
    typer = MinhptxSemanticLabeling(dataset, 200)
    typer.semantic_labeling(sources[:train_size], sources[train_size:], 4)
