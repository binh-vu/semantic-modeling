#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import ujson

from typing import Dict, List, Optional, TYPE_CHECKING

from data_structure import Graph

if TYPE_CHECKING:
    from semantic_modeling.assembling.undirected_graphical_model.model_core import ExampleAnnotator


class Example(object):
    def __init__(self, gold_sm: Graph, pred_sm: Graph, link2label: Dict[int, bool],
                 prime2x: Dict[int, Optional[int]]) -> None:
        self.gold_sm: Graph = gold_sm
        self.pred_sm: Graph = pred_sm

        # mapping from node in pred_sm to gold_sm
        self.prime2x: Dict[int, Optional[int]] = prime2x
        # link id to boolean label
        self.link2label: Dict[int, bool] = link2label or {}

        # for features, this one is set !!dynamically!!
        self.link2features: Dict[int, Dict] = None
        self.node2features: Dict[int, Dict] = None

        # for debugging
        self.training_sources: List[str] = None  # list of training sources name used to build this sample
        # add this to guarantee that we will always get correct sm candidate
        self.example_id: str = None
        self.model_id: str = None

        self.annotator: 'ExampleAnnotator' = None

    @staticmethod
    def generate_example_id(model_id: str, no_example: int, iteration: int):
        return "%s====%02d====%03d" % (model_id, iteration, no_example)

    def get_model_id(self):
        return self.model_id

    def set_meta(self, example_id: str, training_sources: List[str]):
        self.example_id = example_id
        self.model_id = example_id.split("===")[0]
        self.training_sources = training_sources
        return self

    def to_dict(self):
        return {
            "gold_sm": self.gold_sm.to_dict(),
            "pred_sm": self.pred_sm.to_dict(),
            "link2label": self.link2label,
            "prime2x": self.prime2x,
            "training_sources": self.training_sources,
            "example_id": self.example_id
        }

    @staticmethod
    def from_dict(obj: dict) -> 'Example':
        obj['gold_sm'] = Graph.from_dict(obj['gold_sm'])
        obj['pred_sm'] = Graph.from_dict(obj['pred_sm'])
        obj['link2label'] = {int(id): label for id, label in obj['link2label'].items()}
        obj['prime2x'] = {int(k): v for k, v in obj['prime2x'].items()}
        example = Example(obj['gold_sm'], obj['pred_sm'], obj['link2label'], obj['prime2x'])
        example.set_meta(obj['example_id'], obj['training_sources'])

        return example


class OutputExample(object):
    def __init__(self, example_id: str, link2label: Dict[int, bool], log_score: float) -> None:
        self.example_id: str = example_id
        self.link2label: Dict[int, bool] = link2label
        self.log_score = log_score

    @staticmethod
    def from_dict(obj: dict) -> 'OutputExample':
        link2label = {int(k): v for k, v in obj.pop("link2label").items()}
        return OutputExample(link2label=link2label, **obj)


class TrainingArgs(object):
    def __init__(self, n_epoch: int, n_switch: int, mini_batch_size: int, shuffle_mini_batch: bool, manual_seed: int,
                 report_final_loss: bool, optparams: dict, optimizer: str, n_iter_eval: int, parallel_training: bool) -> None:
        self.n_epoch = n_epoch
        self.n_switch = n_switch
        self.n_iter_eval = n_iter_eval
        self.mini_batch_size = mini_batch_size
        self.shuffle_mini_batch = shuffle_mini_batch
        self.manual_seed = manual_seed
        self.report_final_loss = report_final_loss
        self.optparams = optparams
        self.optimizer = optimizer
        self.parallel_training = parallel_training

    @staticmethod
    def parse_shell_args():
        def str2bool(v):
            assert v.lower() in {"true", "false"}
            return v.lower() == "true"

        parser = argparse.ArgumentParser('Train model')
        parser.register("type", "boolean", str2bool)
        parser.add_argument('--n_epoch', type=int, default=40, help='default 40')
        parser.add_argument('--n_switch', type=int, default=10, help='default 10')
        parser.add_argument('--n_iter_eval', type=int, default=5, help='default 5')
        parser.add_argument('--mini_batch_size', type=int, default=200, help='default 200')
        parser.add_argument('--shuffle_mini_batch', type="boolean", default=False, help='Default false')
        parser.add_argument('--manual_seed', type=int, default=120, help='default 120')
        parser.add_argument('--report_final_loss', type="boolean", default=True, help='Default false')
        parser.add_argument('--optparams', type=str, default=ujson.dumps(dict(lr=0.1)), help='default dict(lr=0.1)')
        parser.add_argument('--optimizer', type=str, default='ADAM', help='default ADAM (when using ADAM, u must use amsgrad')
        parser.add_argument('--parallel', type="boolean", default=True, help="Default true")
        args = parser.parse_args()
        args.optparams = ujson.loads(args.optparams)

        return TrainingArgs(args.n_epoch, args.n_switch, args.mini_batch_size,
                            args.shuffle_mini_batch, args.manual_seed, args.report_final_loss,
                            args.optparams, args.optimizer, args.n_iter_eval, args.parallel)

    def to_string(self):
        return f"""
************************************************************
********************* Training parameters ******************

n_epoch             : {self.n_epoch}
n_switch            : {self.n_switch}
n_iter_eval         : {self.n_iter_eval}
mini_batch_size     : {self.mini_batch_size}
shuffle_mini_batch  : {self.shuffle_mini_batch}
manual_seed         : {self.manual_seed}
report_final_loss   : {self.report_final_loss}
optimizer           : {self.optimizer}
optparams           : {self.optparams}
parallel-training   : {self.parallel_training}
************************************************************
"""
