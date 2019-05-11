#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import subprocess
import ujson
from pathlib import Path

import requests
from requests.auth import HTTPDigestAuth

from semantic_modeling.config import config
from semantic_modeling.utilities.serializable import *


def get_shell_args():
    def str2bool(v):
        assert v.lower() in {"true", "false"}
        return v.lower() == "true"

    parser = argparse.ArgumentParser('Assembling experiment')
    parser.register("type", "boolean", str2bool)

    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--exp_dir', type=str, required=True, help='Experiment directory, must be existed before')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_shell_args()

    dataset = args.dataset

    exp_dir = Path(args.exp_dir)
    assert exp_dir.exists()

    kfolds = [file.name for file in exp_dir.iterdir() if file.name.startswith("kfold-") and file.is_dir()]
    eval_header = ["source", "precision", "recall", "f1", 'stype-aac']
    exp_results = {}
    files = []

    for kfold in kfolds:
        eval = deserializeCSV(exp_dir / f"{kfold}.test.csv")
        assert eval[0][0] == 'source' and eval[-1][0] == 'average'
        for r in eval[1:-1]:
            for i in range(1, len(r)):
                r[i] = float(r[i])
        exp_results[f"test-{kfold}"] = [dict(zip(eval[0], r)) for r in eval[1:-1]]
        files += [kfold, f"{kfold}.test.csv"]

        if (exp_dir / f"{kfold}.meta.json").exists():
            files.append(f"{kfold}.meta.json")
        elif (exp_dir / f"{kfold}.meta.yml").exists():
            files.append(f"{kfold}.meta.yml")
        else:
            assert False

    files.append("execution.log")
    files = " ".join(files)
    with open(exp_dir / "commit_id.txt", "r") as f:
        commit_id = f.read().strip()

    if (exp_dir / f"{kfolds[0]}.meta.json").exists():
        with open(exp_dir / f"{kfolds[0]}.meta.json", "r") as f:
            configure = ujson.load(f)
    else:
        assert (exp_dir / f"{kfolds[0]}.meta.yml").exists()
        configure = deserializeYAML(exp_dir / f"{kfolds[0]}.meta.yml")

    auth = HTTPDigestAuth(config.pyexp_api.user, config.pyexp_api['pass'])
    resp = requests.post(config.pyexp_api.host,
                         data=ujson.dumps({
                             "endpoint": "new_run",
                             "problem": "semantic-modeling",
                             "experiment": args.exp_name,
                             "payload": {
                                 "commit_id": commit_id,
                                 "name": args.run_name,
                                 "configure": configure,
                                 "expt_results": exp_results
                             }
                         }), auth=auth)
    assert resp.ok, resp.text
    run_dir = resp.json()['run_dir']

    subprocess.check_call(f"cd {exp_dir}; tar -czf exp.tar.gz {files}", shell=True)
    resp = requests.post(config.pyexp_api.host,
                         files={'file': open(exp_dir / "exp.tar.gz", "rb")},
                         data={'decompress': "true", 'rel_path': run_dir},
                         auth=auth)
    assert resp.ok
