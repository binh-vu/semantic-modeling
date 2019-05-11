import itertools
import pandas
from copy import copy
from pathlib import Path
from typing import *

from data_structure import Graph
from experiments.previous_work.serene_2018.ssd import SSD, SSDAttribute
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models, get_data_tables, get_sampled_data_tables, get_ontology
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeJSON, serializeCSV, serializeJSON
from transformation.models.data_table import DataTable
from transformation.models.table_schema import Schema


def make_ssd(sm: SemanticModel, keys: Set[str], ont: Ontology) -> SSD:
    attrs = {}
    for attr in sm.attrs:
        # new_lbl = attr.label.replace(Schema.PATH_DELIMITER, ".")
        new_lbl = attr.label
        attrs[attr.id] = SSDAttribute(attr.id, new_lbl)
        assert new_lbl in keys

    g = Graph()
    for n in sm.graph.iter_nodes():
        if n.is_data_node():
            label = attrs[n.id].name.encode()
        else:
            label = n.label
        g.add_new_node(n.type, label)
    for e in sm.graph.iter_links():
        g.add_new_link(e.type, e.label, e.source_id, e.target_id)
    return SSD(sm.id, list(attrs.values()), g, ont)


def make_dataset(sm: SemanticModel, tbl: DataTable, ont: Ontology, serene_data_dir: Path, serene_sm_dir: Path):
    def cross_products(row: dict) -> Union[List, Dict]:
        single_fields = {}
        multi_fields = {}
        for key, val in row.items():
            if isinstance(val, dict):
                result = cross_products(val)
                if isinstance(result, dict):
                    single_fields[key] = result
                elif isinstance(result, list):
                    multi_fields[key] = result
                else:
                    raise Exception("Invalid result type: %s" % type(result))
            elif isinstance(val, list):
                multi_fields[key] = val
            else:
                single_fields[key] = val

        if len(multi_fields) == 0:
            return single_fields

        rows = []
        keys, field_values = list(zip(*multi_fields.items()))
        for values in itertools.product(*field_values):
            row = copy(single_fields)
            for i, val in enumerate(values):
                row[keys[i]] = val
            rows.append(row)

        return rows

    def flatten_row(row: dict) -> dict:
        new_row = {}
        for key, val in row.items():
            if isinstance(val, dict):
                for k2, v2 in flatten_row(val).items():
                    new_row[f"{key}{Schema.PATH_DELIMITER}{k2}"] = v2
            else:
                new_row[key] = val
        return new_row

    # flatten a data table
    flatten_rows = []
    for row in tbl.rows:
        new_rows = cross_products(row)
        if isinstance(new_rows, dict):
            new_rows = [new_rows]

        for r in new_rows:
            flatten_rows.append(flatten_row(r))

    # print(DataTable.load_from_rows("", flatten_rows).to_string())
    keys = list(flatten_rows[0].keys())
    values = [
        [r[k] for k in keys]
        for r in flatten_rows
    ]

    serializeCSV([keys] + values, serene_data_dir / f"{sm.id}.csv")
    # create ssds
    ssd = make_ssd(sm, set(keys), ont)
    serializeJSON(ssd.to_dict(), serene_sm_dir / f"{sm.id}.ssd", indent=4)
    # ssd.graph.render(80)


if __name__ == '__main__':
    dataset = "museum_edm"
    ont = get_ontology(dataset)
    # serene_dir = Path(config.as_path()) / "debug" / dataset / "serene"
    serene_dir = Path("/workspace/tmp/serene-python-client/datasets/") / dataset
    serene_data_dir = serene_dir / "dataset"
    serene_sm_dir = serene_dir / "ssd"

    serene_data_dir.mkdir(exist_ok=True, parents=True)
    serene_sm_dir.mkdir(exist_ok=True)

    sms = get_semantic_models(dataset)
    tables = get_sampled_data_tables(dataset)

    for sm, tbl in zip(sms, tables):
        # if not sm.id.startswith("s07"):
        #     continue
        make_dataset(sm, tbl, ont, serene_data_dir, serene_sm_dir)
        # break