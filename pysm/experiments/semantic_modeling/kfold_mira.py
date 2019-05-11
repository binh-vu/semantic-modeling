import argparse
import shutil
from pathlib import Path
from typing import *

from experiments.arg_helper import parse_kfold, str2bool
from semantic_modeling.data_io import get_semantic_models, get_short_train_name
from semantic_modeling.rust_bridge import serialize_rust_input
from semantic_modeling.settings import Settings


def get_shell_args():
    parser = argparse.ArgumentParser('Mira sm experiment')
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


if __name__ == '__main__':
    # HYPER-ARGS
    args = get_shell_args()

    Settings.get_instance(False).semantic_labeling_top_n_stypes = args.styper_top_n_stypes
    Settings.get_instance().semantic_labeling_method = args.styper
    Settings.get_instance().semantic_labeling_simulate_testing = args.styper_simulate_testing

    exp_dir = Path(args.exp_dir)
    assert exp_dir.exists()

    source_models = {sm.id: sm for sm in get_semantic_models(args.dataset)}
    train_sms = [source_models[sid] for sid in args.kfold['train_sm_ids']]
    test_sms = [source_models[sid] for sid in args.kfold['test_sm_ids']]

    workdir = exp_dir / f"kfold-{get_short_train_name(train_sms)}"
    workdir.mkdir(exist_ok=True, parents=True)

    if args.func == 'gen_input':
        Settings.get_instance().log_current_settings()
        
        (workdir / "rust").mkdir(exist_ok=True, parents=True)
        serialize_rust_input(args.dataset, workdir / "rust", train_sms, test_sms, workdir / "rust-input.json")
    else:
        assert args.func == 'handle_output', "Invalid function: %s" % args.func
        config_file = Path(args.config_file)
        assert config_file.exists()
        # copy configuration file
        shutil.copyfile(config_file, exp_dir / f"{workdir.name}.meta{config_file.suffix}")
        # prepare the result
        shutil.copyfile(workdir / "rust" / "result.csv", exp_dir / f"{workdir.name}.test.csv")