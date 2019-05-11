#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas
from collections import defaultdict

import numpy
import ujson
from pathlib import Path
from typing import *

from experiments.arg_helper import get_sm_ids_by_name_range
from experiments.evaluation_metrics import smodel_eval, DataNodeMode
from experiments.previous_work.serene_2018.ssd import SSD
from semantic_modeling.data_io import get_semantic_models, get_ontology
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeJSON
from transformation.models.data_table import DataTable
from transformation.models.table_schema import Schema


def evaluate_serene_outputs(files: List[Path], ont: Ontology, gold_sm: Optional[SemanticModel]=None) -> Union[dict, None]:
    try:
        cor_ssd_file = [file for file in files if file.name.endswith(".cor_ssd.json")][0]
        ssd_file = [file for file in files if file.name.endswith(".ssd.json")][0]
    except Exception as e:
        raise Exception("Invalid : %s" % files[0], e)

    cor_ssd = SSD.from_file(cor_ssd_file, ont).clear_serene_footprint()
    ssd = SSD.from_file(ssd_file, ont)
    chuffed_ssds = []
    for file in files:
        if file.name.find(".chuffed") != -1:
            objs = deserializeJSON(file)
            chuffed_ssds.append([SSD.from_json(obj, ont) for obj in objs])

    if gold_sm is None:
        # SERENE can filter the cor_ssd graph to remove new-semantic types
        gold_graph = cor_ssd.graph
    else:
        gold_graph = gold_sm.graph

    eval_results = {}
    for chuffed_idx, ssds in enumerate(chuffed_ssds):
        eval_results[chuffed_idx] = {}

        if len(ssds) == 0:
            eval_results[chuffed_idx] = {
                'precision': 0,
                'recall': 0,
                'f1': 0
            }
        else:
            ssd = ssds[0]
            # ssd.graph.render()
            result = smodel_eval.f1_precision_recall(gold_graph, ssd.graph, DataNodeMode.NO_TOUCH)
            eval_results[chuffed_idx]['precision'] = result['precision']
            eval_results[chuffed_idx]['recall'] = result['recall']
            eval_results[chuffed_idx]['f1'] = result['f1']

    return eval_results


if __name__ == '__main__':
    dataset = "museum_crm"
    sms = get_semantic_models(dataset)
    sms_index = {sm.id[:3]: sm for sm in sms}
    ont = get_ontology(dataset)
    ont.register_namespace("serene", "http://au.csiro.data61/serene/dev#")

    # get serene output by sms
    kfold_results = []
    stype = "ReImplMinhISWC_False_pat"
    for kfold in ["kfold-s01-s14", "kfold-s15-s28", "kfold-s08-s21"]:
        kfold_sms_prefix = {sm[:3] for sm in get_sm_ids_by_name_range(*kfold.replace("kfold-", "").split("-"), [sm.id for sm in sms])}

        print("==== KFOLD:", kfold, "====")
        serene_output_dir = Path("/workspace/tmp/serene-python-client/datasets/%s/" % dataset) / kfold / f"predicted_{stype}"
        serene_outputs = {}
        if not serene_output_dir.exists():
            print("not existed")
            continue

        for file in sorted(serene_output_dir.iterdir()):
            if not file.name.startswith("s") or file.name.startswith("ser"):
                continue

            prefix = file.name[:3]
            if prefix not in serene_outputs:
                serene_outputs[prefix] = []
            serene_outputs[prefix].append(file)

        methods = defaultdict(lambda: {})
        for prefix, files in sorted(serene_outputs.items(), key=lambda x: x[0]):
            if prefix in kfold_sms_prefix:
                continue

            # if not prefix == "s09":
            #     continue

            # sms_index[prefix].graph.render(80)
            eval_res = evaluate_serene_outputs(files, ont, sms_index[prefix])

            # print(prefix, ujson.dumps(eval_res, indent=4))
            for i, res in eval_res.items():
                methods[i][prefix] = {
                    'precision': res['precision'],
                    'recall': res['recall'],
                    'f1': res['f1'],
                }

        assert len(methods[0]) == 14
        header = ["source"]
        matrix = []
        for method_idx, results in sorted(methods.items(), key=lambda x: x[0]):
            header.append(f"{method_idx}_precision")
            header.append(f"{method_idx}_recall")
            header.append(f"{method_idx}_f1")

        for prefix, o in sorted(methods[0].items(), key=lambda x: x[0]):
            matrix.append([prefix])

        for method_idx, results in sorted(methods.items(), key=lambda x: x[0]):
            for i, (prefix, o) in enumerate(sorted(results.items(), key=lambda x: x[0])):
                matrix[i].append(o['precision'])
                matrix[i].append(o['recall'])
                matrix[i].append(o['f1'])

        print(matrix)
        # print(DataTable.load_from_rows("", matrix).to_string())
        df = pandas.DataFrame(data=matrix, columns=header)
        matrix.append(['average'] + list(df.mean(axis=0)))
        df = pandas.DataFrame(data=matrix, columns=header)
        print(df)

        df.to_csv(serene_output_dir.parent / f"result_{stype}.csv")
        kfold_results.append([kfold] + matrix[-1][1:])

    df = pandas.DataFrame(data=kfold_results, columns=header)
    kfold_results.append(['average'] + list(df.mean(axis=0)))
    df = pandas.DataFrame(data=kfold_results, columns=header)
    df.to_csv(serene_output_dir.parent.parent / f"average_{stype}.csv")
    print(kfold_results[-1])