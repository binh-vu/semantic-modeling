#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse

from experiments.arg_helper import str2bool, parse_kfold


def get_shell_args():
    parser = argparse.ArgumentParser('Oracle experiment')
    parser.register("type", "boolean", str2bool)

    # copied from settings.py
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--func', type=str, required=True, help="Which function to execute: gen_input or handle_output")
    parser.add_argument('--config_file', type=str, help='Location where the configuration file is stored')
    parser.add_argument('--kfold', type=str, required=True,
                        help='kfold json object of {train_sm_ids: [], test_sm_ids: []}. For example: {"train_sm_ids": ["s00-s05"], "test_sm_ids": ["s06-s06"]}')
    parser.add_argument('--styper', type=str, help='kind of semantic type')
    parser.add_argument('--styper_simulate_testing', type='boolean')
    parser.add_argument('--styper_top_n_stypes', type=int, default=1, help='Number of semantic types')
    parser.add_argument('--exp_dir', type=str, help='Experiment directory, must be existed before')

    args = parser.parse_args()
    try:
        assert args.dataset is not None
        args.kfold = parse_kfold(args.dataset, args.kfold)

        if args.func == 'gen_input':
            assert args.styper is not None and args.styper_simulate_testing is not None and args.styper_top_n_stypes is not None
    except AssertionError:
        parser.print_help()
        raise

    return args

