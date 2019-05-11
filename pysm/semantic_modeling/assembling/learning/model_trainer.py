#!/usr/bin/python
# -*- coding: utf-8 -*-

import shutil
import ujson
from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pyutils
from pyutils.list_utils import _

from gmtk.graph_models.factors.template_factor import CachedTemplateFactorConstructor
from gmtk.graph_models.models.model import TemplateLogLinearModel
from gmtk.graph_models.variables.vector_domain import GrowableBinaryVectorDomain
from gmtk.inferences import BeliefPropagation
from gmtk.inferences.inference import InferProb
from gmtk.optimize.example import NegativeLogLikelihoodExample, MAPAssignmentExample, BatchExample, ValueAccumulator, \
    ParallelBatchExample, Tensor1AccumulatorDict
from gmtk.optimize.numerical_gradient import NumericalGradient
from gmtk.optimize.optimizer import PyTorchOptimizer
from gmtk.tensors import DenseTensorFunc
from semantic_modeling.assembling.learning.shared_models import Example, TrainingArgs
from semantic_modeling.assembling.undirected_graphical_model.model_core import ExampleAnnotator
from semantic_modeling.assembling.undirected_graphical_model.templates.triple_template import TripleFactorTemplate
from semantic_modeling.assembling.undirected_graphical_model.model_extra import TensorBoard, evaluate, \
    get_latest_model_id, move_current_files, save_evaluation_result
from semantic_modeling.assembling.undirected_graphical_model.templates.substructure_template import \
    SubstructureFactorTemplate
from semantic_modeling.config import get_logger
from semantic_modeling.utilities.parallel_util import sequential_map
from semantic_modeling.utilities.serializable import serialize, serializeJSON

logger = get_logger('app.persistent.assembling.train_model')


def nll_func(example: NegativeLogLikelihoodExample):
    loss_val_accum = ValueAccumulator()
    example.accumulate_value_and_gradient(loss_val_accum, None)
    return loss_val_accum.get_value()


def train_model(dataset: str, train_sids: List[str], manual_seed: int, train_examples: List[Example],
                test_examples: List[Example], args: TrainingArgs, basedir: Path):
    DenseTensorFunc.manual_seed(manual_seed)

    tf_domain = GrowableBinaryVectorDomain()

    timer = pyutils.progress.Timer().start()
    input_train_examples = train_examples
    input_test_examples = test_examples

    # BUILDING VARIABLES NEEDED FOR THE TRAINING
    example_annotator = ExampleAnnotator(dataset, train_sids, training_examples=train_examples)
    train_examples = sequential_map(example_annotator.annotate, train_examples)
    train_examples = _(train_examples) \
        .imap(example_annotator.example2vars) \
        .submap(partial(example_annotator.build_triple_features, domain=tf_domain))

    pairwise_domain = example_annotator.build_pairwise_domain()
    # Freeze domain now, we've added all feature values observed in training data
    tf_domain.freeze()

    test_examples = sequential_map(example_annotator.annotate, test_examples)
    test_examples = _(test_examples) \
        .imap(example_annotator.example2vars) \
        .submap(partial(example_annotator.build_triple_features, domain=tf_domain))

    # print domain to debug
    logger.info("Preprocessing take %s" % timer.lap().get_total_time())
    # build random variables
    train_graphs = _(train_examples).submap(lambda t: t.label)
    test_graphs = _(test_examples).submap(lambda t: t.label)

    # build models, select inference method
    model = TemplateLogLinearModel([
        TripleFactorTemplate(*TripleFactorTemplate.get_default_args(tf_domain)),
        SubstructureFactorTemplate(
            *SubstructureFactorTemplate.get_default_args(pairwise_domain, example_annotator.get_obj_props())),
        # ExternalModelFactorTemplate(*ExternalModelFactorTemplate.get_default_weights())
    ])
    # or load previous training
    # model_dir = config.fsys.debug.as_path() + "/%s/models/exp_no_2" % dataset
    # model, ___, state_dict = deserialize(model_dir + '/gmtk_model.bin')

    inference = BeliefPropagation.get_constructor(InferProb.MARGINAL)
    map_inference = BeliefPropagation.get_constructor(InferProb.MAP)

    train_nll_examples = _(train_graphs).map(
        lambda vars: NegativeLogLikelihoodExample(vars, model.get_factors(vars), inference))
    train_map_examples = _(train_nll_examples).map(
        lambda example: MAPAssignmentExample.from_nll_example(example, map_inference))
    test_nll_examples = _(test_graphs).map(
        lambda vars: NegativeLogLikelihoodExample(vars, model.get_factors(vars), inference))
    test_map_examples = _(test_nll_examples).map(
        lambda example: MAPAssignmentExample.from_nll_example(example, map_inference))

    # select training method/parameters, and evaluation
    n_epoch = args.n_epoch
    params = args.optparams
    mini_batch_size = args.mini_batch_size
    n_switch = args.n_switch

    global_step = 0
    require_closure = False
    if args.optimizer == 'SGD':
        optimizer = PyTorchOptimizer.SGD(parameters=model.get_parameters(), **params)
    elif args.optimizer == 'ADAM':
        optimizer = PyTorchOptimizer.Adam(parameters=model.get_parameters(), **params)
    elif args.optimizer == 'LBFGS':
        optimizer = PyTorchOptimizer.LBFGS(parameters=model.get_parameters(), **params)
        require_closure = True
    else:
        assert False
    # optimizer.optimizer.load_state_dict(state_dict)

    for template in model.templates:
        if hasattr(template, 'after_update_weights'):
            optimizer.register_on_step(template.after_update_weights)

    logger.info(args.to_string())
    logger.info("Template info: \n%s" % ("\n" % (["\t" + template.get_info() for template in model.templates])))
    logger.info("Train size: %s, Test size: %s", len(train_nll_examples), len(test_nll_examples))

    reporter = TensorBoard(log_dir=basedir)
    # cast to list to keep train_map_examples & train_nll_examples aligned with each other (batch example may shuffle)
    if args.parallel_training:
        batch_nll_example = ParallelBatchExample(list(train_nll_examples), 0)
    else:
        batch_nll_example = BatchExample(list(train_nll_examples), 0)

    # *********************************************** DEBUG CODE
    # for i, triples in enumerate(train_examples):
    #     example = triples[0].example
    #     if example.model_id.startswith("s03") and example.no_sample == 29:
    #         example.pred_sm.render()
    #         render_factor_graph(model.get_factors(train_graphs[i]), train_graphs[i],
    #                     config.fsys.debug.as_path() + "/tmp/factor_graph.pdf")
    #         exit(0)
    #
    # render_factor_graph(train_nll_examples[0].factors, train_nll_examples[0].variables,
    #                     config.fsys.debug.as_path() + "/tmp/factor_graph.pdf")
    #
    # loss_val_accum = ValueAccumulator()
    # gradient_accum = Tensor1AccumulatorDict()
    # for weights in model.get_parameters():
    #     gradient_accum.track_obj(weights, DenseTensorFunc.zeros_like(weights.val))
    # **********************************************************

    progress = pyutils.progress.Progress(n_epoch)
    progress.start()

    if n_switch > 0:
        examples = list(batch_nll_example.split_random(mini_batch_size))
    else:
        examples = [batch_nll_example]

    cm_train, cm_test = None, None
    loss_history = []
    param_hists = []

    for i in range(n_epoch):
        logger.info("Iter %s" % i)

        if i >= n_switch:
            examples = [batch_nll_example]

        if args.shuffle_mini_batch and 0 < i < n_switch:
            examples = batch_nll_example.split_random(mini_batch_size)

        average_loss_val = []
        if not require_closure:
            for example in examples:
                optimizer.zero_grad()
                example.accumulate_value_and_gradient(optimizer.get_value_accumulator(),
                                                      optimizer.get_gradient_accumulator())
                optimizer.average(example.size())

                logger.info("Accum loss: %.10f" % optimizer.get_value_accumulator().get_value())
                average_loss_val.append(optimizer.get_value_accumulator().get_value())

                # *********************************************** DEBUG GRADIENT
                # numerical_gradient = NumericalGradient(1e-5)
                # for j, e in enumerate(example.examples):
                #     print(f"\rExample {j}/{len(example.examples)}", end="", flush=True)
                #     gradient_accum.clear()
                #     loss_val_accum.clear()
                #     e.accumulate_value_and_gradient(loss_val_accum, gradient_accum)
                #     for template in model.templates:
                #         for weights in template.get_weights():
                #             gradient = gradient_accum.get_value(weights)
                #             approx_gradients = numerical_gradient.compute_gradient(weights, lambda: nll_func(e))
                #             try:
                #                 np.testing.assert_almost_equal(gradient.numpy(), approx_gradients.numpy(), 6)
                #             except Exception:
                #                 logger.exception("Incorrect gradient...")
                #                 print(template,  weights.val.tolist())
                #                 print(["%11.8f" % x for x in gradient.tolist()])
                #                 print(["%11.8f" % x for x in approx_gradients.tolist()])
                #                 print(["%11d" % int(np.isclose(x, y, rtol=0, atol=1e-6)) for x, y in zip(gradient, approx_gradients)])
                #
                #                 raise
                # print("\n")
                # **************************************************************

                optimizer.step()
                reporter.loss_val(optimizer.get_value_accumulator().get_value(), global_step)
                global_step += 1
        else:
            for example in examples:
                def closure():
                    optimizer.zero_grad()
                    example.accumulate_value_and_gradient(optimizer.get_value_accumulator(),
                                                          optimizer.get_gradient_accumulator())
                    optimizer.average(example.size())
                    optimizer.copy_grad()
                    return optimizer.get_value_accumulator().get_value()

                optimizer.step(closure)
                logger.info("Accum loss: %.10f" % optimizer.get_value_accumulator().get_value())
                average_loss_val.append(optimizer.get_value_accumulator().get_value())
                reporter.loss_val(optimizer.get_value_accumulator().get_value(), global_step)
                global_step += 1

        if len(average_loss_val) > 1:
            logger.info("Average accum loss: %.10f" % np.average(average_loss_val))

        if optimizer.get_value_accumulator().get_value() < 0:
            break

        if i % args.n_iter_eval == 0 or i == n_epoch - 1:
            cm_train = evaluate(train_map_examples)
            cm_test = evaluate(test_map_examples) or cm_train
            logger.info('train (class_idx=0): %s', cm_train.precision_recall_fbeta(class_idx=0))
            logger.info('train (class_idx=1): %s', cm_train.precision_recall_fbeta(class_idx=1))
            logger.info('test  (class_idx=0): %s', cm_test.precision_recall_fbeta(class_idx=0))
            logger.info('test  (class_idx=1): %s', cm_test.precision_recall_fbeta(class_idx=1))

            reporter.precision_recall_fbeta(cm_train, global_step, group='train')
            reporter.precision_recall_fbeta(cm_test, global_step, group='test')

        loss_history.append(np.average(average_loss_val))
        param_hists.append(model.clone_parameters())
        if len(param_hists) > 3:
            param_hists.pop(0)

        if args.optimizer == "ADAM" and len(loss_history) > 4 and all(
                x - y > 0 for x, y in zip(loss_history[-3:], loss_history[-4:-1])):
            logger.info("Loss increase after 3 epoches. Stop training!")
            break

        progress.finish_one()

    if args.report_final_loss:
        loss_val_accum = ValueAccumulator()
        batch_nll_example.accumulate_value_and_gradient(loss_val_accum, None)
        logger.info("Average accum loss: %.10f" % (loss_val_accum.get_value() / batch_nll_example.size()))

    logger.info("\n\r%s" % progress.summary())
    cm_train.pretty_print("** TRAIN **", precision_recall_fbeta=True, output_stream=logger.info)
    cm_test.pretty_print("** TEST **", precision_recall_fbeta=True, output_stream=logger.info)

    # save model and move everything into another folder for storage
    reporter.close()
    reporter.export(basedir / 'tensorboard_raw.json')

    # clear all cache
    for template in model.templates:
        if isinstance(template, CachedTemplateFactorConstructor):
            template.clear_cache()

    assert len(param_hists) == len(loss_history[-3:])
    min_loss, min_params, min_idx = min(zip(loss_history[-3:], param_hists, [-3, -2, -1]), key=lambda x: x[0])
    logger.info("Select parameters at index: %d. Loss = %s", min_idx, min_loss)
    model.update_parameters(min_params)

    serialize((model, tf_domain, pairwise_domain, optimizer.optimizer.state_dict()), basedir / 'gmtk_model.bin')
    save_evaluation_result(zip(train_map_examples, train_nll_examples), basedir / 'train.output.json')
    save_evaluation_result(zip(test_map_examples, test_nll_examples), basedir / 'test.output.json')
    serializeJSON(input_train_examples, basedir / "train.json")
    serializeJSON(input_test_examples, basedir / "test.json")

    # attempt to copy log file
    try:
        logger.handlers[1].flush()
        shutil.copy(logger.handlers[1].file_handler.baseFilename, str(basedir / "train.log"))
    except:
        logger.exception("Cannot backup log...")

    model_id = get_latest_model_id(basedir) + 1
    move_current_files(basedir, model_id)
    logger.info("Save model %s", model_id)
    return model, tf_domain, pairwise_domain, optimizer.optimizer.state_dict()
