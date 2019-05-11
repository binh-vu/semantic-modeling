import argparse
from pathlib import Path
from typing import *

from experiments.arg_helper import str2bool
from semantic_modeling.utilities.serializable import deserializeYAML, serializeYAML, deserializeJSON, serializeJSON


def get_shell_args():
    parser = argparse.ArgumentParser('Assembling experiment')
    parser.register("type", "boolean", str2bool)

    parser.add_argument('--config_file', type=str, required=True, help="Config file")
    parser.add_argument('--config', type=str, nargs='+', required=True, help='config want to update and its value, separated by colon (:)')
    parser.add_argument('--new_config_file', type=str, default=None, help="Where to store new config file")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_shell_args()
    config_file = Path(args.config_file)

    if config_file.suffix in [".yml", ".yaml"]:
        config = deserializeYAML(args.config_file)
    elif config_file.suffix == ".json":
        config = deserializeJSON(args.config_file)
    else:
        assert "Not support format", args.config_file.suffix

    for conf_string in args.config:
        path, value = conf_string.split(":")
        attrs = path.split(".")

        if value.replace('.', '', 1).isdigit():
            value = float(value)
        if value in {"false", "true"}:
            value = (value == "true")

        current_conf = config
        for path in attrs[:-1]:
            current_conf = current_conf[path]

        current_conf[attrs[-1]] = value

    new_location = args.new_config_file or args.config_file
    if config_file.suffix in [".yml", ".yaml"]:
        serializeYAML(config, new_location)
    elif config_file.suffix == ".json":
        serializeJSON(config, new_location, indent=4)


