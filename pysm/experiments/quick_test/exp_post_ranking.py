import sys
from pathlib import Path

from typing import List, Dict

import numpy

from data_structure import Graph
from experiments.evaluation_metrics import smodel_eval
from semantic_modeling.assembling.autolabel.auto_label import AutoLabel
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import deserializeJSON


class Prediction:

    def __init__(self, content: dict):
        self.sm_id = content['sm_id']
        self.search_history: List[List[Graph]] = [[Graph.from_dict(obj) for obj in preds] for preds in content['search_history']]
        self.search_history_score: List[List[float]] = content['search_history_score']
        self.search_history_map: List[List[List[bool]]] = content['search_history_map']


class Ranking:

    def __init__(self, train_sms: List[SemanticModel], test_sms: List[SemanticModel]) -> None:
        self.train_sms = train_sms
        self.test_sms: Dict[str, SemanticModel] = {sm.id: sm for sm in test_sms}

    def mohsen_coherence(self, g: Graph):
        score = 0
        for sm in train_sms:
            result = smodel_eval.f1_precision_recall(sm.graph, g, 1)
            if result['precision'] > score:
                score = result['precision']

        return score

    def rank(self, predictions: List[Prediction]) -> None:
        average_result = []
        for prediction in predictions:
            graph_w_scores = list(zip(prediction.search_history[-1], prediction.search_history_score[-1]))
            best_graph, best_score = graph_w_scores[0][0], 0
            for graph, score in graph_w_scores:
                coherence = self.mohsen_coherence(graph)
                u = 2 * len(self.test_sms[prediction.sm_id].attrs)
                size_reduction = (u - graph.get_n_nodes()) / (u - sum(1 for _ in graph.iter_data_nodes()))

                rank_score = coherence * 0.5 + size_reduction * 0.5 + score * 0.8
                if rank_score > best_score:
                    best_graph = graph
                    best_score = rank_score

            eval_result = smodel_eval.f1_precision_recall(self.test_sms[prediction.sm_id].graph, best_graph, 0)
            average_result.append(eval_result)

        print("average precision={} recall={} f1={}".format(
            numpy.average([x['precision'] for x in average_result]),
            numpy.average([x['recall'] for x in average_result]),
            numpy.average([x['f1'] for x in average_result])
        ))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        workdir = Path(sys.argv[1])
    else:
        workdir = Path("/workspace/semantic-modeling/debug/museum_crm/run3/")

    kfold_dirs = [dpath for dpath in workdir.iterdir() if dpath.name.startswith("kfold")]
    for kfold_dir in kfold_dirs:
        if not kfold_dir.is_dir():
            continue

        rust_input = deserializeJSON(kfold_dir / "rust-input.json")
        dataset = rust_input['dataset']
        semantic_models = get_semantic_models(dataset)
        train_sms = [semantic_models[i] for i in rust_input['train_sm_idxs']]
        test_sms = [semantic_models[i] for i in rust_input['test_sm_idxs']]

        ranker = Ranking(train_sms, test_sms)
        predictions = [Prediction(obj) for obj in deserializeJSON(kfold_dir / "rust" / "prediction.json")]

        print(kfold_dir.name)
        ranker.rank(predictions)
