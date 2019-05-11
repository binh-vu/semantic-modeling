import shutil
import ujson, sys
from typing import *
from pathlib import Path
from multiprocessing.pool import ThreadPool
from data_structure import *
from semantic_modeling.config import config
from semantic_modeling.data_io import *
from semantic_modeling.assembling.autolabel.auto_label import AutoLabel
from pydebug.colorization import WrappedLink, WrappedOutputLink


def render_prediction(pred, semantic_models: Dict[str, SemanticModel], output_dir: Path):
    sm_id = pred["sm_id"]
    curr_dir = output_dir / sm_id
    curr_dir.mkdir(exist_ok=True, parents=True)

    render_graphs = []

    for iter_no in range(len(pred["search_history"])):
        for i in range(len(pred["search_history"][iter_no])):
            pred_g = Graph.from_dict(pred["search_history"][iter_no][i])
            pred_score = pred["search_history_score"][iter_no][i]
            pred_map = pred["search_history_map"][iter_no][i]

            g = Graph(name=f"{sm_id}-{iter_no}-{i}--score={pred_score}".encode("utf-8"))
            link2label = AutoLabel.auto_label_max_f1(semantic_models[sm_id].graph, pred_g, False)[0]

            for n in pred_g.iter_nodes():
                g.add_new_node(n.type, n.label)
            for e in pred_g.iter_links():
                e_prime = WrappedOutputLink(link2label[e.id], pred_map[e.id])
                g.real_add_new_link(e_prime, e.type, e.label, e.source_id, e.target_id)

            render_graphs.append((g, curr_dir / f"sm_id={sm_id[:3]}---iter={iter_no}--graph={i}.pdf"))
            # (curr_dir / f"iter={iter_no}").mkdir(exist_ok=True, parents=True)

    with ThreadPool() as p:
        p.map(lambda gdn: gdn[0].render2pdf(gdn[1]), render_graphs)


def print_prediction(dataset, rust_dir):
    semantic_models = {sm.id: sm for sm in get_semantic_models(dataset)}
    output_dir = rust_dir / "prediction_viz"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    prediction_file = rust_dir / "prediction.json"

    with open(prediction_file, "r") as f:
        predictions = ujson.load(f)

    for prediction in predictions:
        render_prediction(prediction, semantic_models, output_dir)


if __name__ == "__main__":
    # Load all input
    dataset = "museum_crm"
    if len(sys.argv) == 1:
        workdir = Path("/workspace/semantic-modeling/debug/museum_crm/run_003")
    else:
        workdir = Path(sys.argv[1])
        assert workdir.exists()

    for kfold_dir in workdir.iterdir():
        if kfold_dir.name.startswith("kfold") and kfold_dir.is_dir():
            print_prediction(dataset, kfold_dir / "rust")