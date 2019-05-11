#!/usr/bin/python
# -*- coding: utf-8 -*-
import shutil
from functools import partial
from typing import Dict, List, Union

from pathlib import Path

# import gmtk.config
# gmtk.config.USE_C_EXTENSION = False

import numpy

from data_structure import Graph, graph_to_hashable_string
# from gmtk.tensors import DenseTensorFunc, DType
from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.autolabel.auto_label import AutoLabel
from semantic_modeling.assembling.constructing_semantic_model import Tracker, beam_search
from semantic_modeling.settings import Settings
from semantic_modeling.assembling.graph_explorer_builder import GraphExplorerBuilder
from semantic_modeling.assembling.learning.model_trainer import train_model
from semantic_modeling.assembling.learning.shared_models import TrainingArgs, Example
from semantic_modeling.assembling.ont_graph import get_ont_graph
from semantic_modeling.assembling.triple_adviser import EmpiricalTripleAdviser
from semantic_modeling.assembling.undirected_graphical_model.model import Model
from semantic_modeling.assembling.undirected_graphical_model.search_discovery import PGMBeamSearchArgs, \
    PGMStartSearchNode, PGMSearchNode, discovering_func, filter_unlikely_graph
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.config import config, get_logger
from semantic_modeling.data_io import get_ontology, get_semantic_models, get_short_train_name
from semantic_modeling.karma.karma_node import KarmaSemanticType
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import serializeJSON, deserializeJSON
from semantic_modeling.utilities.parallel_util import get_pool, AsyncResult


def custom_search_discovery(search_nodes: List[Union[PGMStartSearchNode, PGMSearchNode]],
                            args: PGMBeamSearchArgs) -> List[PGMSearchNode]:
    max_tolerant_errors = 3
    max_candidates = 30

    # explorer more nodes, and add those nodes back to the store as training data
    beam_width = args.beam_width
    args.beam_width = 10000
    next_nodes = discovering_func(search_nodes, args)
    args.beam_width = beam_width

    # do a filter out here, if the next nodes having more than 3 errors, stop it
    # also do an oracle selection to make sure it doesn't go wrong!
    controlled_next_nodes = []
    for node in next_nodes:
        link2label = AutoLabel.auto_label_max_f1_no_ambiguous(args.gold_sm, node.get_value().graph, False)[0]
        if link2label is None or len(link2label) - sum(link2label.values()) >= max_tolerant_errors:
            # ignore this examples either there are more than one possible mapping, or contain so many errors
            continue
        controlled_next_nodes.append(node)

    # if len(controlled_next_nodes) <= args.beam_width:
    #     # don't need to add more correct examples, as it should generate correct examples
    #     pass

    # FIRST OPTION: still need to have some correct examples + wild-structures
    # train_nodes = controlled_next_nodes[:15] + args._tmp_random_state.choice(controlled_next_nodes[15:], size=max_candidates, replace=False)
    # SECOND OPTION: sampling based on the predicted scores
    # THIRD OPTION: mixed
    if len(controlled_next_nodes) < max_candidates:
        train_nodes = controlled_next_nodes
    else:
        n_priority_boarding = 10
        p = numpy.asarray([search_node.get_score() for search_node in controlled_next_nodes[n_priority_boarding:]])
        train_nodes = controlled_next_nodes[:n_priority_boarding] + args._tmp_random_state.choice(
            controlled_next_nodes[n_priority_boarding:],
            size=max_candidates - n_priority_boarding,
            replace=False,
            p=p / p.sum()).tolist()

    if not hasattr(args, "_tmp_tracker_for_storing_search_discovery_nodes"):
        # random sampling instead of top 30
        args._tmp_tracker_for_storing_search_discovery_nodes = train_nodes
    else:
        args._tmp_tracker_for_storing_search_discovery_nodes += train_nodes

    return controlled_next_nodes[:args.beam_width]


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

    model = Model(*model_bundle)
    args = PGMBeamSearchArgs(
        test_sm.id,
        custom_search_discovery,
        Tracker(track_search_nodes=False),
        partial(model.predict_sm_probs, test_sm.id, train_source_ids),
        graph_explorer_builder,
        early_terminate_func=None,
        beam_width=settings.training_beam_width,
        gold_sm=test_sm.graph,
        source_attributes=test_sm.attrs,
        pre_filter_func=filter_unlikely_graph,
    )
    started_node = PGMStartSearchNode(args.get_and_increment_id(), args,
                                      [a.label.encode('utf-8') for a in test_sm.attrs])

    args._tmp_random_state = numpy.random.RandomState(Settings.get_instance().random_seed)

    results: List[PGMSearchNode] = beam_search(
        [started_node],
        beam_width=settings.training_beam_width,
        n_results=settings.searching_n_explore_result,
        args=args)

    candidate_sms = {}
    for search_node in args._tmp_tracker_for_storing_search_discovery_nodes:
        g = search_node.get_value().graph
        candidate_sms[graph_to_hashable_string(g)] = g

    for search_node in results:
        g = search_node.get_value().graph
        candidate_sms[graph_to_hashable_string(g)] = g

    return candidate_sms


def make_example(sm: SemanticModel, g: Graph, example_id, train_sids: List[str] = None) -> Example:
    settings = Settings.get_instance()
    if settings.auto_labeling_method == Settings.ALGO_AUTO_LBL_MAX_F1:
        link2label, prime2x = AutoLabel.auto_label_max_f1(sm.graph, g, False)[:2]
        example = Example(sm.graph, g, link2label, prime2x)
        example.set_meta(example_id, train_sids)

        return example
    assert False


def generate_data(model: Model, dataset: str, train_sms: List[SemanticModel], discover_sources: List[SemanticModel],
                  n_iter):
    data = {}
    stat = Statistic.get_instance(train_sms)
    train_sids = [sm.id for sm in train_sms]
    model_bundle = (model.dataset, model.model, model.tf_domain, model.pairwise_domain)

    with get_pool(Settings.get_instance().parallel_n_process) as pool:
        results = []
        for source in discover_sources:
            result: AsyncResult[Dict[bytes, Graph]] = pool.apply_async(
                generate_candidate_sm, (dataset, source, stat, model_bundle, train_sids))
            results.append(result)

        for source, result in zip(discover_sources, results):
            candidate_sms = result.get()
            for i, key in enumerate(candidate_sms):
                candidate_sms[key] = make_example(source, candidate_sms[key],
                                                  Example.generate_example_id(source.id, i, n_iter), train_sids)

            data[source.id] = candidate_sms

    return data


def online_learning(model: Model,
                    dataset: str,
                    train_sms: List[SemanticModel],
                    discover_sources: List[SemanticModel],
                    output_dir: Path,
                    training_args,
                    iter_range=(1, 3)):
    data: Dict[str, Dict[bytes, Example]] = {sm.id: {} for sm in discover_sources}
    discover_sids = {sm.id for sm in discover_sources}
    ignore_sids = set()  # those should not include in the discovery_helper process because of no new sources
    logger = get_logger("app")
    (output_dir / "examples").mkdir(exist_ok=True, parents=True)

    # default should have ground-truth
    for sm in discover_sources:
        data[sm.id][graph_to_hashable_string(sm.graph)] = make_example(sm, sm.graph,
                                                                       Example.generate_example_id(sm.id, 0, 0),
                                                                       [sm.id for sm in train_sms])

    for n_iter in range(*iter_range):
        logger.info("==================================> Iter: %s", n_iter)
        new_data = generate_data(model, dataset, train_sms, discover_sources, n_iter)
        for sm in discover_sources:
            if sm.id in ignore_sids:
                continue

            new_candidate_sms = [key for key in new_data[sm.id] if key not in data[sm.id]]
            if len(new_candidate_sms) == 0:
                # no new candidate sms
                logger.info("No new candidate for source: %s", sm.id)
                ignore_sids.add(sm.id)
            else:
                for key in new_candidate_sms:
                    data[sm.id][key] = new_data[sm.id][key]

        train_examples = [example for sm in train_sms if sm.id in discover_sids for example in data[sm.id].values()]
        train_examples.sort(key=lambda e: e.example_id)

        serializeJSON(train_examples, output_dir / "examples" / f"train.{n_iter}.json")
        shutil.copyfile(output_dir / "examples" / f"train.{n_iter}.json", output_dir / "examples" / f"train.json")

        raw_model, tf_domain, pairwise_domain, __ = train_model(
            dataset, [sm.id for sm in train_sms], 120, train_examples, [], training_args, output_dir / "models")
        model = Model(dataset, raw_model, tf_domain, pairwise_domain)

    return model


def build_test_data(model: Model, dataset: str, train_sms: List[SemanticModel], discover_sources: List[SemanticModel],
                    output_dir: Path, n_iter):
    data: Dict[str, Dict[bytes, Example]] = {sm.id: {} for sm in discover_sources}
    discover_sids = {sm.id for sm in discover_sources}
    (output_dir / "examples").mkdir(exist_ok=True, parents=True)

    # default should have ground-truth
    for sm in discover_sources:
        data[sm.id][graph_to_hashable_string(sm.graph)] = make_example(sm, sm.graph,
                                                                       Example.generate_example_id(sm.id, 0, 0),
                                                                       [sm.id for sm in train_sms])

    new_data = generate_data(model, dataset, train_sms, discover_sources, 1)
    for sm in discover_sources:
        new_candidate_sms = [key for key in new_data[sm.id] if key not in data[sm.id]]
        for key in new_candidate_sms:
            data[sm.id][key] = new_data[sm.id][key]

    test_examples = [example for sid in discover_sids for example in data[sid].values()]
    test_examples.sort(key=lambda e: e.example_id)

    serializeJSON(test_examples, output_dir / "examples" / f"test.{n_iter}.json")


def create_default_model(dataset: str, train_sms: List[SemanticModel], training_args, basedir: Path) -> Model:
    train_examples = []
    for sm in train_sms:
        example = Example(sm.graph, sm.graph, {e.id: True
                                               for e in sm.graph.iter_links()},
                          {n.id: n.id
                           for n in sm.graph.iter_nodes()})
        example.set_meta(Example.generate_example_id(sm.id, 0, 0), [sm.id for sm in train_sms])
        train_examples.append(example)

    raw_model, tf_domain, pairwise_domain, __ = train_model(dataset, [sm.id for sm in train_sms], 120, train_examples,
                                                            [], training_args, basedir)
    return Model(dataset, raw_model, tf_domain, pairwise_domain)


if __name__ == '__main__':
    from semantic_modeling.assembling.learning.evaluate import predict_sm
    # DenseTensorFunc.set_default_type(DType.Double)

    Settings.get_instance(False).parallel_gmtk_n_threads = 12
    Settings.get_instance().log_current_settings()

    dataset = "museum_edm"
    source_models = get_semantic_models(dataset)
    train_sms = source_models[:6]
    train_sm_ids = [sm.id for sm in train_sms]
    test_sms = [sm for sm in source_models if sm.id not in train_sm_ids]

    workdir = Path(config.fsys.debug.as_path()) / dataset / "main_experiments" / get_short_train_name(train_sms)
    workdir.mkdir(exist_ok=True, parents=True)

    create_semantic_typer(dataset, train_sms).semantic_labeling(
        train_sms, test_sms, top_n=Settings.get_instance().semantic_labeling_top_n_stypes, eval_train=True)

    Settings.get_instance().parallel_gmtk_n_threads = 6
    Settings.get_instance().parallel_n_process = 2
    training_args = TrainingArgs.parse_shell_args()

    # model = create_default_model(dataset, train_sms, training_args, workdir / "models")
    # model = Model.from_file(dataset, workdir / "models" / "exp_no_0")
    # model = online_learning(model, dataset, train_sms, train_sms, workdir, training_args, iter_range=(1, 3))

    # model = Model.from_file(dataset, workdir / "models" / "exp_no_2")
    # build_test_data(model, dataset, train_sms, test_sms, workdir, 2)

    # predictions = predict_sm(model, dataset, [sm.id for sm in train_sms], test_sms, model_dir)
    # evaluate(test_sms, predictions, model_dir)
    #

    train_examples = deserializeJSON(workdir / "examples" / f"train.2.json", Class=Example)
    test_examples = deserializeJSON(workdir / "examples" / f"test.json", Class=Example)
    # test_examples = train_examples
    args = TrainingArgs.parse_shell_args()
    args.parallel_training = True
    args.n_switch = 19
    args.n_epoch = 22
    args.mini_batch_size = 200
    args.shuffle_mini_batch = True
    # args.n_iter_eval = 50
    # args.optparams = {"lr": 0.005, "amsgrad": True}
    # args.optimizer = 'LBFGS'
    # args.optparams = {"lr": 0.1}
    args.optparams = {"lr": 0.1, "amsgrad": True}
    model_bundle = train_model(dataset, [sm.id for sm in train_sms], 120, train_examples, test_examples, args,
                               workdir / "models")
    # model = Model(dataset, *model_bundle[:-1])
