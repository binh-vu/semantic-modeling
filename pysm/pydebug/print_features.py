import sys
import ujson
from pathlib import Path

from typing import List

from data_structure import *
from semantic_modeling.utilities.serializable import deserializeJSON, serializeCSV, serializeJSON


def print_primary_keys(input, output_file: Path):
    with open(output_file, "w") as f:
        f.write(ujson.dumps(input['feature_primary_keys'], indent=4))


def print_stypes(input, output_file: Path):
    with open(output_file, "w") as f:
        for sm in input["semantic_models"]:
            f.write("Source: %s\n" % sm["name"])
            for attr in sm["attrs"]:
                g = Graph.from_dict(sm["graph"])
                correct_st = g.get_node_by_id(attr["id"]).get_first_incoming_link()
                f.write("\t- attr: %s -- correct type (%s --- %s)\n" % (
                attr["label"], correct_st.get_source_node().label.decode(), correct_st.label.decode()))
                for i, st in enumerate(attr["semantic_types"]):
                    if (st["domain"], st["type"]) == (
                    correct_st.get_source_node().label.decode(), correct_st.label.decode()):
                        bullet = f"({i + 1})"
                    else:
                        bullet = f"{i + 1}"
                    f.write(f"\t\t{bullet}. %20s --- %15s: %.3f\n" % (st["domain"], st["type"], st["confidence_score"]))


def print_triple_features(features_file_content: dict, train_output_file: Path, test_output_file: Path):
    def json2csv(rows: List[dict]):
        if len(rows) == 0:
            return []

        header = list(rows[0].keys())
        if "provenance" in header:
            header.remove("provenance")
        header = ['provenance'] + header
        data = [header]
        for row in rows:
            data.append([row[k] for k in header])

        return data

    serializeCSV(json2csv(features_file_content['train_examples']), train_output_file)
    serializeCSV(json2csv(features_file_content['test_examples']), test_output_file)


def print_cooccurrence(features_file_content: dict, output_file: Path):
    serializeJSON(features_file_content['cooccurrence'], output_file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        workdir = Path(sys.argv[1])
    else:
        workdir = Path("/workspace/semantic-modeling/debug/museum_crm/run")

    for kfold_dir in workdir.iterdir():
        if kfold_dir.name.startswith("kfold") and kfold_dir.is_dir():
            input = kfold_dir / "rust-input.json"
            output_dir = kfold_dir / "features"
            output_dir.mkdir(exist_ok=True)

            with open(input, "r") as f:
                input = ujson.load(f)

            print_primary_keys(input, output_dir / "pk.txt")
            print_stypes(input, output_dir / "stypes.txt")

            if (kfold_dir / "rust" / "examples.debug.features.json").exists():
                features = deserializeJSON(kfold_dir / "rust" / "examples.debug.features.json")
                print_triple_features(features, output_dir / "triple_features.train.csv",
                                      output_dir / "triple_features.test.csv")
                print_cooccurrence(features, output_dir / "cooccurrence.json")
