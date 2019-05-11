#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import operator
from collections import namedtuple
from pathlib import Path
from typing import List, Dict, Union, Tuple, Iterable, Optional

from pyutils.list_utils import _
from tensorboardX import SummaryWriter

from data_structure import GraphNode, GraphLink, Graph, GraphLinkType, GraphNodeType
from gmtk.eval import ConfusionMatrix, Evaluation
from gmtk.graph_models.factors.factor import Factor
from gmtk.graph_models.models.model import LogLinearModel
from gmtk.graph_models.variables.variable import Variable
from gmtk.graph_models.variables.vector_domain import BinaryVectorValue
from gmtk.graph_models.variables.vector_variable import LabeledBooleanVectorVariable
from gmtk.optimize.example import MAPAssignmentExample, NegativeLogLikelihoodExample
from semantic_modeling.assembling.learning.shared_models import Example, OutputExample
from semantic_modeling.assembling.undirected_graphical_model.model_core import TripleLabel
from semantic_modeling.config import config

from semantic_modeling.utilities.serializable import serializeJSON, deserializeJSON


def discretize_value_uniformly(value: float, min_val: float, max_val: float, nBin: int) -> str:
    size = (max_val - min_val) / nBin
    binNo = math.ceil(max(value - min_val, 1e-6) / size) - 1
    return "bin(%.3f,%.3f)" % (min_val + size * binNo, min_val + size * (binNo + 1))


def load_data(dataset):
    data_dir = Path(config.fsys.debug.as_path()) / dataset / "training_workflow" / "examples_generator" / "i0"
    # train_examples: List[Example] = deserializeJSON(data_dir / "train.small.json", Class=Example)
    # test_examples: List[Example] = deserializeJSON(data_dir / "test.small.json", Class=Example)

    train_examples: List[Example] = deserializeJSON(data_dir / "train.json", Class=Example)
    test_examples: List[Example] = deserializeJSON(data_dir / "test.json", Class=Example)

    # TODO: uncomment below to create small dataset to debug
    # train_examples = [e for e in train_examples if e.model_id.startswith('s03')]
    # train_examples = train_examples[:100]
    # test_examples = test_examples[:100]
    # serializeJSON(train_examples, data_dir / "train.small.json")
    # serializeJSON(test_examples, data_dir / "test.small.json")

    return train_examples, test_examples


def evaluate(map_examples) -> Optional[ConfusionMatrix]:
    if len(map_examples) == 0:
        return None
    return _(map_examples).imap(lambda e: (e.get_map_assignment(), e.get_target_assignment())).imap(
        lambda e: Evaluation.get_confusion_matrix(*e)).reduce(operator.add)


def save_evaluation_result(map_and_nll_examples: Iterable[Tuple[MAPAssignmentExample, NegativeLogLikelihoodExample]],
                           fpath: Union[Path, str]) -> None:
    outputs = []
    confusion_matrixes = []

    for map_example, nll_example in map_and_nll_examples:
        real_example: Example = nll_example.variables[0].triple.example
        map_assignment: Dict[TripleLabel, BinaryVectorValue[bool]] = map_example.get_map_assignment()

        link2labels = {}
        for var, val in map_assignment.items():
            link2labels[var.triple.link.id] = val.val

        desired_assignment = {var: var.domain.encode_value(True) for var in nll_example.variables}
        log_prob = sum(f.score_assignment(desired_assignment)
                       for f in nll_example.factors) - nll_example.inference.logZ()

        output = OutputExample(real_example.example_id, link2labels, log_prob)
        outputs.append(output)
        confusion_matrixes.append(Evaluation.get_confusion_matrix(map_assignment, nll_example.target_assignment))

    serializeJSON(outputs, fpath)


def TensorBoard(log_dir: Union[str, Path]):
    writer = SummaryWriter(log_dir)

    def close():
        writer.close()

    def loss_val(loss_val, global_step, group='train'):
        writer.add_scalar("%s/loss_val" % group, loss_val, global_step)

    def precision_recall_fbeta(cm: ConfusionMatrix, global_step, group='train'):
        argss = cm.all_precision_recall_fbeta(1)
        for name, args in zip(cm.class_names, argss):
            writer.add_scalar("%s/lbl=%s/precision" % (group, name), args[0], global_step)
            writer.add_scalar("%s/lbl=%s/recall" % (group, name), args[1], global_step)
            writer.add_scalar("%s/lbl=%s/fbeta" % (group, name), args[2], global_step)

    def export(fpath: str):
        writer.export_scalars_to_json(fpath)

    props = locals()
    return namedtuple('TensorBoard', props.keys())(*props.values())


def move_current_files(workdir, model_id: int):
    modeldir = workdir / ("exp_no_%s" % model_id)
    modeldir.mkdir(exist_ok=True, parents=True)
    for item in workdir.iterdir():
        if item.is_dir():
            continue

        item.rename(modeldir / item.name)


def get_latest_model_id(workdir) -> int:
    model_ids = _(workdir.iterdir()).ifilter(lambda e: e.is_dir() and e.name.startswith('exp_no_')).map(
        lambda e: int(e.name.replace('exp_no_', '')))
    if len(model_ids) == 0:
        return -1
    return max(model_ids)


def render_factor_graph(model_or_factors: Union[LogLinearModel, List[Factor]], vars: List[TripleLabel], fpath: str):
    if isinstance(model_or_factors, LogLinearModel):
        factors = model_or_factors.get_factors(vars)
    else:
        factors = model_or_factors

    def get_fnode_lbl(fnode: Union[TripleLabel, Factor]) -> bytes:
        if isinstance(fnode, Factor):
            label = fnode.__class__.__name__
        else:
            s = fnode.triple.link.get_source_node()
            t = fnode.triple.link.get_target_node()
            label = "%s:%s--%s:%s" % (s.id, s.label.decode('utf-8'), t.id, t.label.decode('utf-8'))

        return label.encode('utf-8')

    class Node(GraphNode):
        def __init__(self, fnode: Union[TripleLabel, Factor]) -> None:
            super().__init__()
            self.fnode = fnode

        def get_dot_format(self, max_text_width: int):
            label = self.get_printed_label(max_text_width).encode('unicode_escape').decode()
            if isinstance(self.fnode, Variable):
                return '"%s"[style="filled",color="white",fillcolor="gold",label="%s"];' % (self.id, label)

            return '"%s"[shape="plaintext",style="filled",fillcolor="lightgray",label="%s"];' % (self.id, label)

    class Link(GraphLink):
        var2factor = "var2factor"
        var2var = "var2var"

        def __init__(self, link_type: str) -> None:
            super().__init__()
            self.link_type = link_type

        def get_dot_format(self, max_text_width: int):
            label = self.get_printed_label(max_text_width).encode('unicode_escape').decode()
            if self.link_type == Link.var2factor:
                return '"%s" -> "%s"[dir=none,color="brown",fontcolor="black",label="%s"];' % (self.source_id,
                                                                                               self.target_id, label)
            return '"%s" -> "%s"[color="brown",style="dashed",fontcolor="black",label="%s"];' % (self.source_id,
                                                                                                 self.target_id, label)

    """Render factor graph for debugging"""
    g = Graph()

    # build graphs
    fnode2id: Dict[Union[Variable, Factor], int] = _(vars, factors).enumerate().imap(lambda v: (v[1], v[0])).todict()
    _(vars,
      factors).forall(lambda fnode: g.real_add_new_node(Node(fnode), GraphNodeType.CLASS_NODE, get_fnode_lbl(fnode)))

    for factor in factors:
        for var in factor.unobserved_variables:
            g.real_add_new_link(Link(Link.var2factor), GraphLinkType.UNSPECIFIED, b"", fnode2id[var], fnode2id[factor])
    for var in vars:
        if var.triple.parent is not None:
            g.real_add_new_link(
                Link(Link.var2var), GraphLinkType.UNSPECIFIED, b"", fnode2id[var.triple.parent.label], fnode2id[var])

    for var in vars:
        var.myid = "%s: %s" % (fnode2id[var], g.get_node_by_id(fnode2id[var]).label)
    for factor in factors:
        factor.myid = fnode2id[factor]

    g.render2pdf(fpath)


def get_numbered_link_label(link_label: str, number: int) -> str:
    """Number a link"""
    return "%s:_%d" % (link_label, number)


def get_unnumbered_link_label(numbered_link_label: str) -> str:
    """Get original link"""
    return numbered_link_label[:numbered_link_label.rfind(":_")]
