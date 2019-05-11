import ujson, sys
from typing import *
from pathlib import Path
from itertools import chain
from data_structure import *
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_short_train_name

def satisfy_forward_prop(prop, node_or_edge: Union[GraphNode, GraphLink], idx=0):
    if idx == len(prop):
        return True

    if isinstance(node_or_edge, GraphNode):
        if prop[idx] == node_or_edge.label or prop[idx] == b'*':
            satisfied = False
            for e in node_or_edge.iter_outgoing_links():
                satisfied = satisfied or satisfy_forward_prop(prop, e, idx + 1)

            return satisfied
        else:
            return False
    else:
        assert isinstance(node_or_edge, GraphLink)
        if prop[idx] == node_or_edge.label or prop[idx] == b'*':
            return satisfy_forward_prop(prop, node_or_edge.get_target_node(), idx + 1)
        else:
            return False

if __name__ == "__main__":
    # Load all input
    dataset = "museum_crm"
    workdir = Path(f"/workspace/semantic-modeling/debug/{dataset}/run3/kfold-s11-to-s29")
    semantic_models = get_semantic_models(dataset)

    train_file = workdir / "rust/examples.train.elimination.json"
    test_file = workdir / "rust/examples.test.elimination.json"
    # test_file = None

    node_lbl = b"crm:E52_Time-Span"
    forward_props = [
        (b"crm:P82a_begin_of_the_begin",),
        (b"crm:P82a_begin_of_the_begin",),
        # (b"schema:name",),
        # (b"schema:name",),
    ]

    with open(train_file, "r") as f:
        train_examples = ujson.load(f)

    if test_file is not None:
        with open(test_file, "r") as f:
            test_examples = ujson.load(f)
    else:
        test_examples = []

    for no, example in enumerate(chain(train_examples, test_examples)):
        pred_sm = Graph.from_dict(example["graph"])
        for n in pred_sm.iter_nodes_by_label(node_lbl):
            used_props = set()
            for e in n.iter_outgoing_links():
                for i, prop in enumerate(forward_props):
                    if i in used_props:
                        continue

                    if satisfy_forward_prop(prop, e):
                        used_props.add(i)
                        break

            if len(used_props) == len(forward_props):
                if no <= len(train_examples):
                    print("train, example:", no, "sm", semantic_models[example['sm_idx']].id)
                else:
                    print("test, example:", no - len(train_examples), "sm", semantic_models[example['sm_idx']].id)