#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict, OrderedDict
from itertools import chain
from pathlib import Path

from typing import Dict, Tuple, List, Union, Optional

import numpy
import pandas
from pyutils.list_utils import _
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from data_structure import GraphNode, Graph
from semantic_modeling.assembling.learning.shared_models import Example

from semantic_modeling.assembling.weak_models.multi_val_predicate import MultiValuePredicate
from semantic_modeling.config import config, get_logger
from semantic_modeling.data_io import get_semantic_models, get_cache_dir
from semantic_modeling.utilities.serializable import serialize, deserialize


def get_merged_cost(nodeA: GraphNode, nodeB: GraphNode, multi_val_predicate: MultiValuePredicate):
    """"determining the merged cost from node B to node A. This one would be asymmetric"""

    # make this asymmetric function
    if nodeA.n_outgoing_links > nodeB.n_outgoing_links:
        return 1e6

    # TODO: how about the case that # their nodes is equal ??

    pseudo_outgoing_links = defaultdict(lambda: 0)
    for link in chain(nodeA.iter_outgoing_links(), nodeB.iter_outgoing_links()):
        pseudo_outgoing_links[link.label] += 1

    total_cost = 0
    for link_lbl, link_no in pseudo_outgoing_links.items():
        cost = multi_val_predicate.compute_prob(link_lbl, link_no)
        # this one likes a product of prob, then higher is better (likely to merge), then we should negative it
        if cost is None:
            cost = 0  # assume that this link is always fine, then log(cost) = 0
        else:
            cost = -numpy.log(max(cost, 1e-6))
            # cost = -max(cost, 1e-6)
        total_cost += cost
    return total_cost


class NodeProb(object):

    logger = get_logger("app.assembling.weak_models.node_prob")

    def __init__(self, example_annotator: 'ExampleAnnotator', load_classifier: bool=False):
        self.example_annotator = example_annotator
        self.multival_predicate = example_annotator.multival_predicate

        if load_classifier:
            retrain = example_annotator.training_examples is not None
            self.scaler, self.classifier = self.get_classifier(retrain=retrain, train_examples=example_annotator.training_examples)
        else:
            self.scaler, self.classifier = None, None

    def feature_extraction(self, graph: Graph, stype_score: Dict[int, Optional[float]]):
        node2features = {}
        for node in graph.iter_class_nodes():
            prob_data_nodes = _(node.iter_outgoing_links()) \
                .imap(lambda x: x.get_target_node()) \
                .ifilter(lambda x: x.is_data_node()) \
                .reduce(lambda a, b: a + (stype_score[b.id] or 0), 0)

            similar_nodes = graph.iter_nodes_by_label(node.label)
            minimum_merged_cost = min(
                (get_merged_cost(node, similar_node, self.multival_predicate) for similar_node in similar_nodes))

            node2features[node.id] = [
                ('prob_data_nodes', prob_data_nodes),
                ('minimum_merged_cost', minimum_merged_cost)
            ]
        return node2features

    def compute_prob(self, node2features):
        X = numpy.asarray([
            [p[1] for p in features]
            for features in node2features.values()
        ])

        self.scaler.transform(X)
        y_pred = self.classifier.predict_proba(X)[:, 1]
        return {nid: y_pred[i] for i, nid in enumerate(node2features.keys())}

    def get_classifier(self, retrain: bool, train_examples: List[Example]):
        # TODO: implement this properly, currently, we have to train and save manually
        cached_file = get_cache_dir(self.example_annotator.dataset, list(self.example_annotator.train_source_ids)) / "weak_models" / "node_prob_classifier.pkl"
        if not cached_file.exists() or retrain:
            self.logger.debug("Retrain new model")
            raw_X_train = make_data(self, train_examples)
            classifier = LogisticRegression(fit_intercept=True)

            X_train = numpy.asarray([list(features.values())[1:] for features in raw_X_train])
            X_train, y_train = X_train[:, :-1], [int(x) for x in X_train[:, -1]]

            scaler = StandardScaler().fit(X_train)
            scaler.transform(X_train)

            try:
                classifier.fit(X_train, y_train)
            except ValueError as e:
                assert str(e).startswith("This solver needs samples of at least 2 classes in the data")
                # this should be at a starter phase when we don't have any data but use ground-truth to build
                X_train = numpy.vstack([X_train, [0, 0]])
                y_train.append(0)
                classifier.fit(X_train, y_train)

            cached_file.parent.mkdir(exist_ok=True, parents=True)
            serialize((scaler, classifier), cached_file)
            return scaler, classifier

        return deserialize(cached_file)


def make_data(node_prob, examples: List[Example]):
    """Use to create training data"""
    X = []
    for example in examples:
        stype_score = node_prob.example_annotator.get_stype_score(example)
        node2features = node_prob.feature_extraction(example.pred_sm, stype_score)
        for node in example.pred_sm.iter_class_nodes():
            features = OrderedDict([('provenance', '%s:%s' % (example.example_id, node.id))])
            features.update(node2features[node.id])
            features['label'] = example.prime2x[node.id] is not None
            X.append(features)
    return X


if __name__ == '__main__':
    from semantic_modeling.assembling.training_workflow.mod_interface import ExampleLabelingFileInterface
    from semantic_modeling.assembling.training_workflow.training_manager import WorkflowSettings
    from semantic_modeling.assembling.undirected_graphical_model.model_core import ExampleAnnotator

    dataset = "museum_crm"
    source_models = get_semantic_models(dataset)
    train_source_ids = [sm.id for sm in source_models[:12]]

    # load model
    workdir = Path(config.fsys.debug.as_path()) / dataset / "training_workflow"
    # noinspection PyTypeChecker
    settings = WorkflowSettings(
        dataset,
        max_iter=1,
        workdir=workdir,
        train_source_ids=train_source_ids,
        test_source_ids=None,
        model_trainer_args=None,
        scenario=None)

    annotator = ExampleAnnotator(dataset, train_source_ids, load_circular_dependency=False)
    node_prob = NodeProb(annotator)

    # make training data
    train_examples, test_examples = ExampleLabelingFileInterface(settings.workdir, set()).read_examples_at_iter(0)
    raw_X_train = make_data(node_prob, train_examples)
    raw_X_test = make_data(node_prob, test_examples)

    output_dir = Path(config.fsys.debug.as_path()) / dataset / "weak_models" / "node_prob"
    output_dir.mkdir(exist_ok=True, parents=True)
    pandas.DataFrame(raw_X_train).to_csv(str(output_dir / "features.train.csv"), index=False)
    pandas.DataFrame(raw_X_test).to_csv(str(output_dir / "features.test.csv"), index=False)

    # create classifier to train
    classifier = LogisticRegression(fit_intercept=True)

    X_train = numpy.asarray([list(features.values())[1:] for features in raw_X_train])
    X_test = numpy.asarray([list(features.values())[1:] for features in raw_X_test])
    X_train, y_train = X_train[:, :-1], [int(x) for x in X_train[:, -1]]
    X_test, y_test = X_test[:, :-1], [int(x) for x in X_test[:, -1]]

    scaler = StandardScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)

    classifier.fit(X_train, list(y_train))
    print("Weights", classifier.coef_, classifier.intercept_)

    # report performance
    y_train_pred = classifier.predict(X_train)
    res = precision_recall_fscore_support(y_train, y_train_pred)
    print('Train', res)
    res = precision_recall_fscore_support(y_test, classifier.predict(X_test))
    print('Test', res)

    # save classifier
    serialize((scaler, classifier), output_dir / "classifier.pkl")

    print(classifier.predict([[0.938, 0]]))
