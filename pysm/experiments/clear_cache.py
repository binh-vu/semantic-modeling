import argparse
import shutil

from semantic_modeling.data_io import get_cache_dir


def clear_cache(dataset: str) -> None:
    # only clear cache which are generated for different training models
    cache_dir = get_cache_dir(dataset)
    for item in cache_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)


def get_shell_args():
    def str2bool(v):
        assert v.lower() in {"true", "false"}, f"Got {v.lower()}"
        return v.lower() == "true"

    parser = argparse.ArgumentParser('Clear all cache')
    parser.register("type", "boolean", str2bool)

    parser.add_argument('--dataset', type=str, required=True, default="museum_edm", help="Dataset name")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_shell_args()
    clear_cache(args.dataset)
