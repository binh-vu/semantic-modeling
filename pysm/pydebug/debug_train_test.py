import ujson, sys
from typing import *
import pyutils
from pathlib import Path
from multiprocessing.pool import ThreadPool
from data_structure import *
from semantic_modeling.config import config
from semantic_modeling.data_io import *
from semantic_modeling.assembling.autolabel.auto_label import AutoLabel
from pydebug.colorization import WrappedLink, WrappedOutputLink
from semantic_modeling.utilities.serializable import deserializeJSON


def to_color_graph(sid: str, g: Graph, link2label, pred_link2label) -> Graph:
    new_g = Graph(name=sid.encode())
    for n in g.iter_nodes():
        new_g.add_new_node(n.type, n.label)
    
    if pred_link2label is None:
        for e in g.iter_links():
            e_prime = WrappedLink(link2label[e.id])
            new_g.real_add_new_link(e_prime, e.type, e.label, e.source_id, e.target_id)
    else:
        for e in g.iter_links():
            e_prime = WrappedOutputLink(link2label[e.id], pred_link2label[e.id])
            new_g.real_add_new_link(e_prime, e.type, e.label, e.source_id, e.target_id)
    return new_g


def render_examples(examples, output_dir: Path):
    sms = get_semantic_models(dataset)
    sm_ids = {sms[example['sm_idx']].id for example in examples}
    for sm_id in sm_ids:
        (output_dir / sm_id).mkdir(exist_ok=True, parents=True)

    render_graphs = [
        (
            i,
            to_color_graph(
                f"{sms[example['sm_idx']].id}-{i}",
                Graph.from_dict(example["graph"]), 
                example["label"]["edge2label"],
                example.get("map_link2label", None)
            ),
            sms[example['sm_idx']].id
        )
        for i, example in enumerate(examples)
    ]

    with ThreadPool() as p:
        p.map(lambda igs: igs[1].render2pdf(output_dir / igs[2] / f"example_no_{igs[0]}.pdf"), render_graphs)


if __name__ == "__main__":
    # Load all input
    dataset = sys.argv[1]
    workdir = Path(sys.argv[2])
    train_or_test_file = workdir / sys.argv[3]

    assert workdir.exists()
    assert train_or_test_file.exists()

    semantic_models = {sm.id: sm for sm in get_semantic_models(dataset)}
    timer = pyutils.progress.Timer().start()

    examples = deserializeJSON(train_or_test_file)
    # for example, map_example in zip(train_examples, train_map_examples):
    #     example["map_link2label"] = map_example

    if 'train' in str(train_or_test_file).lower():
        render_examples(examples, train_or_test_file.parent / "train_viz")
    elif 'test' in str(train_or_test_file).lower():
        render_examples(examples, train_or_test_file.parent / "test_viz")
    else:
        print("Cannot detect type is train or test. Exit!!")
        exit(0)
    print("Render examples: %s" % timer.lap().get_total_time(), flush=True)