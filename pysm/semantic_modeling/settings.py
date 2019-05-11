#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import ujson
from typing import Dict, Tuple, List, Union, Optional

from semantic_modeling.config import get_logger
from semantic_modeling.karma.semantic_model import SemanticModel


class Settings(object):

    logger = get_logger("app.assembling.settings")
    instance = None

    # ####################################################################
    # Semantic Labeling constant
    ReImplMinhISWC = "ReImplMinhISWC"
    MohsenJWS      = "MohsenJWS"
    OracleSL       = "OracleSL"

    # Searching constant
    ALGO_ES_DISABLE  = "NoEarlyStopping"
    ALGO_ES_MIN_PROB = "MinProb"

    # Auto-labeling constant
    ALGO_AUTO_LBL_MAX_F1              = "AUTO_LBL_MAX_F1"
    ALGO_AUTO_LBL_PRESERVED_STRUCTURE = "AUTO_LBL_PRESERVED_STRUCTURE"
    # ####################################################################

    def __init__(self):
        # ####################################################################
        # General arguments
        self.random_seed: int = 120
        self.n_samples: int   = 1000

        # ####################################################################
        # semantic labeling arguments
        self.semantic_labeling_method: str            = Settings.ReImplMinhISWC
        self.semantic_labeling_top_n_stypes: int      = 4
        self.semantic_labeling_simulate_testing: bool = False

        # ####################################################################
        # auto labeling arguments
        self.auto_labeling_method: str = Settings.ALGO_AUTO_LBL_MAX_F1

        # ####################################################################
        # weak models arguments
        self.data_constraint_guess_datetime_threshold: float = 0.5
        self.data_constraint_valid_threshold: float          = 0.95
        self.data_constraint_n_comparison_samples: int       = 150

        # ####################################################################
        # graphical model arguments
        self.mrf_max_n_props                = 10
        self.mrf_max_n_duplications         = 5
        self.mrf_max_n_duplication_types    = 4

        # ####################################################################
        # searching arguments
        self.training_beam_width: int                    = 10
        self.searching_beam_width: int                   = 10
        self.searching_max_data_node_hop: int            = 2
        self.searching_max_class_node_hop: int           = 2
        self.searching_n_explore_result                  = 5
        self.searching_triple_adviser_max_candidate: int = 15

        self.searching_early_stopping_method: str        = Settings.ALGO_ES_DISABLE
        self.searching_early_stopping_minimum_expected_accuracy   = 0
        self.searching_early_stopping_min_prob_args: Tuple[float] = (0.01,)

        # ####################################################################
        # parallels
        self.parallel_gmtk_n_threads: int = 8
        self.parallel_n_process: int      = 4
        self.parallel_n_annotators: int   = 8
        self.max_n_tasks: int             = 80  # tune this parameter if its consume lots of memory

    def log_current_settings(self):
        self.logger.info("Current settings: %s", self.to_string())

    def set_setting(self, key: str, value, log_change: bool=True):
        assert key in self.__dict__
        self.__dict__[key] = value
        if log_change:
            self.log_current_settings()

    @staticmethod
    def get_instance(print_settings: bool=True) -> 'Settings':
        if Settings.instance is None:
            Settings.instance = Settings()
            if print_settings:
                Settings.instance.log_current_settings()

        return Settings.instance

    @staticmethod
    def parse_shell_args(print_settings: bool=True):
        def str2bool(v):
            assert v.lower() in {"true", "false"}
            return v.lower() == "true"

        parser = argparse.ArgumentParser('Settings')
        parser.register("type", "boolean", str2bool)
        parser.add_argument('--random_seed', type=int, default=120, help='default 120')
        parser.add_argument('--n_samples', type=int, default=1000, help='default 1000')

        parser.add_argument('--semantic_labeling_method', type=str, default='ReImplMinhISWC', help='can be OracleSL, ReImplMinhISWC and MohsenISWC, default ReImplMinhISWC')
        parser.add_argument('--semantic_labeling_top_n_stypes', type=int, default=4, help='Default is top 4')
        parser.add_argument('--semantic_labeling_simulate_testing', type='boolean', default=False, help='Default is False')

        parser.add_argument('--auto_labeling_method', type=str, default='AUTO_LBL_MAX_F1', help='can be AUTO_LBL_MAX_F1 and AUTO_LBL_PRESERVED_STRUCTURE (default AUTO_LBL_MAX_F1)')
        
        parser.add_argument('--data_constraint_guess_datetime_threshold', type=int, default=0.5, help='default 0.5')
        parser.add_argument('--data_constraint_valid_threshold', type=int, default=0.95, help='default is 0.95')
        parser.add_argument('--data_constraint_n_comparison_samples', type=int, default=150, help='default is 150')

        parser.add_argument('--training_beam_width', type=int, default=10, help='default 10')
        parser.add_argument('--searching_beam_width', type=int, default=10, help='default 10')
        parser.add_argument('--searching_max_data_node_hop', type=int, default=2, help='default 2')
        parser.add_argument('--searching_max_class_node_hop', type=int, default=2, help='default 2')
        parser.add_argument('--searching_n_explore_result', type=int, default=5, help='default 5')
        parser.add_argument('--searching_triple_adviser_max_candidate', type=int, default=15, help='default 15')
        parser.add_argument('--searching_early_stopping_method', type=str, default='NoEarlyStopping', help='can be NoEarlyStopping or MinProb (default NoEarlyStopping)')
        parser.add_argument('--searching_early_stopping_minimum_expected_accuracy', type=int, default=0, help='default 0')
        parser.add_argument('--searching_early_stopping_min_prob_args', type=str, default="[0.01]", help='default is [0.01]')

        parser.add_argument('--parallel_gmtk_n_threads', type=int, default=8, help='default is 8 threads')
        parser.add_argument('--parallel_n_process', type=int, default=4, help='default is 4 processes')
        parser.add_argument('--parallel_n_annotators', type=int, default=8, help='default is 8')
        parser.add_argument('--max_n_tasks', type=int, default=80, help='default is 80')

        args = parser.parse_args()
        args.searching_early_stopping_min_prob_args = ujson.loads(args.searching_early_stopping_min_prob_args)

        assert args.semantic_labeling_method in {Settings.ReImplMinhISWC, Settings.MohsenJWS, Settings.OracleSL}
        assert args.auto_labeling_method in {Settings.ALGO_AUTO_LBL_MAX_F1, Settings.ALGO_AUTO_LBL_PRESERVED_STRUCTURE}
        assert args.searching_early_stopping_method in {Settings.ALGO_ES_DISABLE, Settings.ALGO_ES_MIN_PROB}

        Settings.get_instance(False)
        settings = Settings.instance

        settings.random_seed = args.random_seed
        settings.n_samples = args.n_samples
        settings.semantic_labeling_method = args.semantic_labeling_method
        settings.semantic_labeling_top_n_stypes = args.semantic_labeling_top_n_stypes
        settings.semantic_labeling_simulate_testing = args.semantic_labeling_simulate_testing

        settings.auto_labeling_method = args.auto_labeling_method
        settings.data_constraint_guess_datetime_threshold = args.data_constraint_guess_datetime_threshold
        settings.data_constraint_valid_threshold = args.data_constraint_valid_threshold
        settings.data_constraint_n_comparison_samples = args.data_constraint_n_comparison_samples
        settings.searching_beam_width = args.searching_beam_width
        settings.searching_max_data_node_hop = args.searching_max_data_node_hop
        settings.searching_max_class_node_hop = args.searching_max_class_node_hop
        settings.searching_n_explore_result = args.searching_n_explore_result
        settings.searching_triple_adviser_max_candidate = args.searching_triple_adviser_max_candidate
        settings.searching_early_stopping_method = args.searching_early_stopping_method
        settings.searching_early_stopping_minimum_expected_accuracy = args.searching_early_stopping_minimum_expected_accuracy
        settings.searching_early_stopping_min_prob_args = args.searching_early_stopping_min_prob_args
        settings.parallel_gmtk_n_threads = args.parallel_gmtk_n_threads
        settings.parallel_n_process = args.parallel_n_process
        settings.parallel_n_annotators = args.parallel_n_annotators
        settings.max_n_tasks = args.max_n_tasks

        if print_settings:
            settings.log_current_settings()

        return settings

    def to_string(self):
        return f"""
************************************************************
********************* Global settings **********************

*** General arguments

random_seed : {self.random_seed}
n_samples   : {self.n_samples}

*** Semantic labeling arguments

semantic_labeling_method            : {self.semantic_labeling_method}
semantic_labeling_top_n_stypes      : {self.semantic_labeling_top_n_stypes}
semantic_labeling_simulate_testing  : {self.semantic_labeling_simulate_testing}

*** Auto labeling arguments

self.auto_labeling_method : {self.auto_labeling_method}

*** Weak models arguments

data_constraint_valid_threshold          : {self.data_constraint_valid_threshold}
data_constraint_guess_datetime_threshold : {self.data_constraint_guess_datetime_threshold}
data_constraint_n_comparison_samples     : {self.data_constraint_n_comparison_samples}

*** MRF arguments
mrf_max_n_props                : {self.mrf_max_n_props}
mrf_max_n_duplications         : {self.mrf_max_n_duplications}
mrf_max_n_duplication_types    : {self.mrf_max_n_duplication_types}

*** Searching arguments

training_beam_width                                : {self.training_beam_width}
searching_beam_width                               : {self.searching_beam_width}
searching_max_data_node_hop                        : {self.searching_max_data_node_hop}
searching_max_class_node_hop                       : {self.searching_max_class_node_hop}
searching_n_explore_result                         : {self.searching_n_explore_result}
searching_triple_adviser_max_candidate             : {self.searching_triple_adviser_max_candidate}
searching_early_stopping_method                    : {self.searching_early_stopping_method}
searching_early_stopping_minimum_expected_accuracy : {self.searching_early_stopping_minimum_expected_accuracy}
searching_early_stopping_min_prob_args             : {self.searching_early_stopping_min_prob_args}

*** Parallel arguments

parallel_gmtk_n_threads   : {self.parallel_gmtk_n_threads}
parallel_n_process        : {self.parallel_n_process}
parallel_n_annotators      : {self.parallel_n_annotators}
max_n_tasks               : {self.max_n_tasks}
************************************************************
"""
