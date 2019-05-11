#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Optional

from pyutils.list_utils import _

from gmtk.graph_models.models.model import LogLinearModel
from semantic_modeling.config import config
from gmtk.inferences.belief_propagation import BeliefPropagation
from gmtk.inferences.inference import InferProb
from gmtk.optimize.example import NegativeLogLikelihoodExample, MAPAssignmentExample
from semantic_modeling.methods.assembling.undirected_graphical_model.model_extra import load_data, \
    save_evaluation_result, TripleLabel, example2vars
from semantic_modeling.methods.assembling.undirected_graphical_model.model_train import build_triple_features, TripleFactorTemplate, SubstructureFactorTemplate
from semantic_modeling.methods.assembling.undirected_graphical_model.tree_build_data import Example
from semantic_modeling.utilities.serializable import deserialize


def get_example_detail(model_prefix, no_sample, link_id, model: LogLinearModel, graphs: List[List[TripleLabel]]):
    for graph in graphs:
        example = graph[0].triple.example
        if not example.model_id.startswith(model_prefix) or example.no_sample != no_sample:
            continue

        var = _(graph).first(lambda v: v.triple.link.id == link_id)
        assignment = {v: v.get_label_value() for v in graph}
        factors = model.get_factors(graph)
        print(var.triple.link.label)

        for lbl in [False, True]:
            assignment[var] = var.domain.encode_value(lbl)
            print("When value of variable is: ", assignment[var].val)
            score = 0.0
            for factor in factors:
                if factor.touch(var):
                    score += factor.score_assignment(assignment)
                    if isinstance(factor, TripleFactorTemplate.TripleFactor):
                        features = factor.assignment2features(assignment)
                        print('\t. Factor features: ', [(var.triple.features.domain.get_category(idx), features[idx])
                                                        for idx in var.triple.features.get_active_index()])
                    else:
                        print("\t .Factor features: ", factor.assignment2features(assignment).tolist())
            print("\t .Score = ", score)
        break


if __name__ == '__main__':
    data_source = 'museum_crm'
    model_dir = config.fsys.debug.as_path() + "/%s/models/exp_no_10" % data_source
    model, tf_domain = deserialize(model_dir + '/gmtk_model.bin')

    # print out top-K features
    topK = 20
    class_idx = 0
    assert len(model.templates) == 2
    triple_factor = model.templates[0]
    triple_factor_weights = triple_factor.weights.view(2, -1)
    features = [(tf_domain.get_category(i), x, triple_factor_weights[1, i])
                for i, x in enumerate(triple_factor_weights[0, :])]
    features.sort(key=lambda x: x[1], reverse=True)
    for f in features:
        print(f)
    substructure = model.templates[1].weights
    print(substructure)

    # re-populate data and output evaluation
    train_examples, test_examples = load_data(data_source)
    _(train_examples, test_examples).iflatten().forall(lambda x: build_triple_features(x, tf_domain))
    train_graphs = _(train_examples).submap(lambda t: t.label)
    test_graphs = _(test_examples).submap(lambda t: t.label)

    # get detail explanation of one link
    get_example_detail('s00', 0, 'L014', model, test_graphs)

    # NOTE: uncommment code below to run full-evaluation

    # inference = BeliefPropagation.get_constructor(InferProb.MARGINAL)  # , max_iter=5)
    # map_inference = BeliefPropagation.get_constructor(InferProb.MAP)  # , max_iter=5)
    #
    # train_nll_examples = _(train_graphs).map(lambda vars: NegativeLogLikelihoodExample(vars, model, inference))
    # train_map_examples = _(train_nll_examples).map(lambda example: MAPAssignmentExample(example, map_inference))
    # test_nll_examples = _(test_graphs).map(lambda vars: NegativeLogLikelihoodExample(vars, model, inference))
    # test_map_examples = _(test_nll_examples).map(lambda example: MAPAssignmentExample(example, map_inference))
    #
    # cm_train = save_evaluation_result(train_map_examples, model_dir + '/training.output.json')
    # cm_test = save_evaluation_result(test_map_examples, model_dir + '/testing.output.json')
    #
    # cm_train.pretty_print("** TRAIN **")
    # cm_test.pretty_print("** TEST  **")
