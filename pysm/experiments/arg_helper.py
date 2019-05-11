#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import ujson
from typing import Dict, Tuple, List, Set, Union, Optional, Any

from semantic_modeling.data_io import get_semantic_models, get_short_train_name


def str2bool(v):
    assert v.lower() in {"true", "false"}
    return v.lower() == "true"


def get_sm_ids_by_name_range(start_name, end_name, sm_ids):
    start_idx, end_idx = None, None
    for i, sid in enumerate(sm_ids):
        if sid.startswith(start_name):
            assert start_idx is None
            start_idx = i
        if sid.startswith(end_name):
            assert end_idx is None
            end_idx = i

    assert start_idx <= end_idx
    return sm_ids[start_idx:end_idx + 1]  # inclusive


def parse_kfold(dataset, kfold_arg):
    # support some shorthand like: {train_sm_ids: ["s08-s21"], test_sm_ids: ["s01-s07", "s22-s28"]}, inclusive
    kfold_arg = ujson.loads(kfold_arg)
    sm_ids = sorted([sm.id for sm in get_semantic_models(dataset)])
    train_sm_ids = []
    for shorthand in kfold_arg['train_sm_ids']:
        start, end = shorthand.split("-")
        train_sm_ids += get_sm_ids_by_name_range(start, end, sm_ids)
    test_sm_ids = []
    for shorthand in kfold_arg['test_sm_ids']:
        start, end = shorthand.split("-")
        test_sm_ids += get_sm_ids_by_name_range(start, end, sm_ids)
    kfold_arg['train_sm_ids'] = train_sm_ids
    kfold_arg['test_sm_ids'] = test_sm_ids

    return kfold_arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Arg helper')

    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--kfold', type=str, required=True,
                        help='kfold json object of {train_sm_ids: [], test_sm_ids: []}. For example: {"train_sm_ids": ["s00-s05"], "test_sm_ids": ["s06-s06"]}')
    parser.add_argument('--func', type=str, required=True, help='Command to run')

    args = parser.parse_args()
    args.kfold = parse_kfold(args.dataset, args.kfold)

    source_models = {sm.id: sm for sm in get_semantic_models(args.dataset)}
    train_sms = [source_models[sid] for sid in args.kfold['train_sm_ids']]

    if args.func == 'get_short_train_name':
        print(get_short_train_name(train_sms))
    else:
        parser.print_help()
        raise ValueError(f"Invalid function: {args.func}")

