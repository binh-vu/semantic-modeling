#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import shutil, numpy as np
from functools import partial
from multiprocessing.pool import Pool
from typing import Dict, Tuple, List, Iterable

from pathlib import Path

from data_structure import Graph
from pydebug.colorization import colorize_prediction
from experiments.evaluation_metrics import smodel_eval, DataNodeMode
from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.autolabel.auto_label import AutoLabel
from semantic_modeling.assembling.constructing_semantic_model import Tracker, beam_search
from semantic_modeling.settings import Settings
from semantic_modeling.assembling.graph_explorer_builder import GraphExplorerBuilder
from semantic_modeling.assembling.learning.online_learning import make_example
from semantic_modeling.assembling.learning.shared_models import Example
from semantic_modeling.assembling.ont_graph import get_ont_graph
from semantic_modeling.assembling.triple_adviser import EmpiricalTripleAdviser
from semantic_modeling.assembling.undirected_graphical_model.model import Model, ParallelAnnotator
from semantic_modeling.assembling.undirected_graphical_model.search_discovery import PGMBeamSearchArgs, \
    PGMStartSearchNode, PGMSearchNode, discovering_func, filter_unlikely_graph
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.config import config, get_logger
from semantic_modeling.data_io import get_ontology, get_semantic_models, get_short_train_name
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import serializeJSON, serializeCSV, serialize, deserialize, \
    deserializeJSON
from semantic_modeling.utilities.parallel_util import get_pool, AsyncResult

class EarlyStopping(object):

    logger = get_logger("app.assembling.training_workflow.early_stopping")

    def __init__(self) -> None:
        self.prev_max_score = 1

    def early_stopping(self, n_iter, search_nodes: Iterable[PGMSearchNode]):
        # return True
        # return n_iter >= 4
        try:
            search_node = next(iter(search_nodes))
        except StopIteration:
            return True

        current_score = search_node.get_score()
        if self.prev_max_score - current_score > 0.3 and current_score < 0.3:
            return True
        self.prev_max_score = current_score
        return False


def generate_candidate_sm(dataset: str, test_sm: SemanticModel, stat: Statistic, model_bundle, train_source_ids):
    # generate candidate
    ont = get_ontology(dataset)
    ont_graph = get_ont_graph(dataset)
    settings = Settings.get_instance()

    dnodes: Dict[bytes, List[KarmaSemanticType]] = {
        attr.label.encode('utf-8'): attr.semantic_types
        for attr in test_sm.attrs
    }

    ota = EmpiricalTripleAdviser(ont_graph, ont, stat.p_triple, settings.searching_triple_adviser_max_candidate)
    graph_explorer_builder = GraphExplorerBuilder(
        ota,
        max_data_node_hop=settings.searching_max_data_node_hop,
        max_class_node_hop=settings.searching_max_class_node_hop)

    for attr, semantic_types in dnodes.items():
        ota.add_data_node(attr, semantic_types)

    early_stopping = EarlyStopping()
    model = Model(*model_bundle)
    args = PGMBeamSearchArgs(
        test_sm.id,
        discovering_func,
        Tracker(track_search_nodes=True),
        partial(model.predict_sm_probs, test_sm.id, train_source_ids),
        graph_explorer_builder,
        # early_terminate_func=early_stopping.early_stopping,
        early_terminate_func=None,
        beam_width=settings.searching_beam_width,
        gold_sm=test_sm.graph,
        source_attributes=test_sm.attrs,
        pre_filter_func=filter_unlikely_graph,
    )
    started_nodes = [PGMStartSearchNode(args.get_and_increment_id(), args,
                                      [a.label.encode('utf-8') for a in test_sm.attrs])]

    results: List[PGMSearchNode] = beam_search(
        started_nodes,
        beam_width=settings.searching_beam_width,
        n_results=settings.searching_n_explore_result,
        args=args)

    # *****************************************************************************************************************'
    # DEBUG CODE
    output_dir = Path(config.fsys.debug.as_path() + "/tmp/final/")
    # for search_node in args.tracker.list_search_nodes:
    #     search_node.beam_search_args = None
    # serialize(args.tracker.list_search_nodes, output_dir / "search_nodes2.pkl")

    # for file in output_dir.iterdir():
    #     if file.is_dir():
    #         shutil.rmtree(file)
    #     else:
    #         os.remove(file)
    #
    # for i, search_nodes in enumerate(args.tracker.list_search_nodes):
    #     if len(search_nodes) == 0:
    #         continue
    #
    #     sub_output_dir = output_dir / str(i)
    #     sub_output_dir.mkdir(exist_ok=True, parents=True)
    #
    #     for j, r in enumerate(search_nodes[:30]):
    #         pred_sm = r.get_value().graph
    #         pred_sm.set_name(str(r.get_score()).encode('utf-8'))
    #
    #         g = colorize_prediction(pred_sm, AutoLabel.auto_label_max_f1(test_sm.graph, pred_sm, False)[0])
    #         g.render2img(sub_output_dir / f"{j}.png")
    #         serialize(pred_sm, sub_output_dir / f"{j}.pkl")
    #
    # sub_output_dir = output_dir / "result"
    # sub_output_dir.mkdir(exist_ok=True, parents=True)
    #
    # for i, r in enumerate(results):
    #     pred_sm = r.get_value().graph
    #     pred_sm.set_name(str(r.get_score()).encode('utf-8'))
    #
    #     g = colorize_prediction(pred_sm, AutoLabel.auto_label_max_f1(test_sm.graph, pred_sm, False)[0])
    #     g.render2img(sub_output_dir / f"{i}.png")
    #     serialize(pred_sm, sub_output_dir / f"{i}.pkl")
    #
    # # STEP 4: report performance
    print(f"{test_sm.id}: Performance at prev iter:",
          smodel_eval.f1_precision_recall(test_sm.graph, args.tracker.list_search_nodes[-1][0].get_value().graph,
                                          DataNodeMode.NO_TOUCH))
    print(f"{test_sm.id}: Performance at final iter:",
          smodel_eval.f1_precision_recall(test_sm.graph, results[0].get_value().graph, DataNodeMode.NO_TOUCH))
    # *****************************************************************************************************************'
    performances = []
    for iter_no, search_nodes in enumerate(args.tracker.list_search_nodes):
        if len(search_nodes) == 0:
            continue

        x = smodel_eval.f1_precision_recall(test_sm.graph, search_nodes[0].get_value().graph, DataNodeMode.NO_TOUCH)
        performances.append((iter_no, search_nodes[0].get_score(), x['precision'], x['recall'], x['f1']))

    x = smodel_eval.f1_precision_recall(test_sm.graph, results[0].get_value().graph, DataNodeMode.NO_TOUCH)
    performances.append((len(performances), results[0].get_score(), x['precision'], x['recall'], x['f1']))

    pred_sms = [(search_node.get_score(), search_node.get_value().graph) for search_node in results]
    search_history = [[n.get_value().graph for n in search_nodes] for search_nodes in args.tracker.list_search_nodes]
    search_history.append([n.get_value().graph for n in results])

    return pred_sms, performances, search_history


def predict_sm(model: Model, dataset: str, train_sms: List[SemanticModel], evaluate_sms: List[SemanticModel], workdir):
    train_sids = [sm.id for sm in train_sms]
    predictions: Dict[str, Graph] = {}
    stat = Statistic.get_instance(train_sms)

    model_bundle = (model.dataset, model.model, model.tf_domain, model.pairwise_domain)
    search_performance_history = {}
    search_history = {}

    with get_pool(Settings.get_instance().parallel_n_process) as pool:
        results = []
        for sm in evaluate_sms:
            result = pool.apply_async(generate_candidate_sm, (dataset, sm, stat, model_bundle, train_sids))
            results.append(result)

        pred_sms: Tuple[List[Tuple[float, Graph]], List[Tuple[int, float, float, float, float]], List[List[Graph]]]
        for sm, result in zip(evaluate_sms, results):
            pred_sms = result.get()
            predictions[sm.id] = pred_sms[0][0][1]
            search_performance_history[sm.id] = pred_sms[1]
            search_history[sm.id] = pred_sms[2]

    serializeJSON({sid: o.to_dict() for sid, o in predictions.items()}, workdir / "predicted_sms.json")
    serializeJSON(search_performance_history, workdir / "search_performance_history.json", indent=4)
    serializeJSON({
        sid: [[o.to_dict() for o in os] for os in oss]
        for sid, oss in search_history.items()}, workdir / "search_history.json")
    return predictions


def evaluate(evaluate_sms: List[SemanticModel], predictions: Dict[str, Graph], workdir):
    eval_hist = [["source", "precision", "recall", "f1", 'stype-aac']]
    for sm in evaluate_sms:
        eval_result = smodel_eval.f1_precision_recall(sm.graph, predictions[sm.id], DataNodeMode.NO_TOUCH)
        eval_hist.append([sm.id, eval_result["precision"], eval_result["recall"], eval_result["f1"], smodel_eval.stype_acc(sm.graph, predictions[sm.id])])

    eval_hist.append([
        'average',
        np.average([x[1] for x in eval_hist[1:]]),
        np.average([x[2] for x in eval_hist[1:]]),
        np.average([x[3] for x in eval_hist[1:]]),
        np.average([x[4] for x in eval_hist[1:]])
    ])

    serializeCSV(eval_hist, workdir / f"evaluation_result.csv")
    return eval_hist


def make_test_from_prediction(train_sms: List[SemanticModel], evaluate_sms: List[SemanticModel], workdir: Path, model_dir: Path):
    search_history: Dict[str, List[List[dict]]] = deserializeJSON(model_dir / "search_history.json")
    evaluate_sms = {sm.id: sm for sm in evaluate_sms}
    train_sm_ids = [sm.id for sm in train_sms]

    test_examples = []
    for sid in search_history:
        for i, gs in enumerate(search_history[sid]):
            for j, g in enumerate(gs):
                eid = Example.generate_example_id(sid, j, i)
                example = make_example(evaluate_sms[sid], Graph.from_dict(g), eid, train_sm_ids)
                test_examples.append(example)

    serializeJSON(test_examples, workdir / "examples" / "test.json")
    return test_examples


if __name__ == '__main__':
    dataset = "museum_edm"
    Settings.get_instance(False).parallel_n_process = 6
    Settings.get_instance().max_n_tasks = 160
    Settings.get_instance().semantic_labeling_top_n_stypes = 4
    Settings.get_instance().searching_beam_width = 5
    Settings.get_instance().log_current_settings()

    source_models = get_semantic_models(dataset)
    train_sms = source_models[:6]
    test_sms = [sm for sm in source_models if sm not in train_sms]

    workdir = Path(config.fsys.debug.as_path()) / dataset / "main_experiments" / get_short_train_name(train_sms)
    workdir.mkdir(exist_ok=True, parents=True)

    create_semantic_typer(dataset, train_sms).semantic_labeling(
        train_sms,
        test_sms,
        top_n=Settings.get_instance().semantic_labeling_top_n_stypes,
        eval_train=True)

    model_dir = workdir / "models" / "exp_no_3"
    model = Model.from_file(dataset, model_dir)
    evaluate_sms = list(filter(lambda sm: True or sm.id.startswith("s21"), test_sms))
    # predictions = predict_sm(model, dataset, train_sms, evaluate_sms, model_dir)
    # evaluate(evaluate_sms, predictions, model_dir)

    make_test_from_prediction(train_sms, evaluate_sms, workdir, model_dir)