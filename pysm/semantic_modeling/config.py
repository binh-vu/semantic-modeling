#!/usr/bin/python
# -*- coding: utf-8 -*-
import logging
import os
from logging import Logger

import pkg_resources
import pyutils
from pyutils.config_utils import Configuration, load_config

current_file_path = pkg_resources.resource_filename("semantic_modeling", "config.py")
PACKAGE_DIR = os.path.abspath(os.path.join(current_file_path, "../../"))
config: Configuration = load_config(os.path.join(PACKAGE_DIR, '../config.yml'))
logger_config: Configuration = load_config(os.path.join(PACKAGE_DIR, 'logging.yml'))


def get_logger(name) -> Logger:
    return pyutils.logging.Logger.get_instance(
        logger_config, init_logging=False).get_logger(name)
