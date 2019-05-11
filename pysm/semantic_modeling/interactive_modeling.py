#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *

from pathlib import Path

from semantic_labeling import SemanticTyper
from semantic_modeling.config import config
from semantic_modeling.data_io import get_ontology, get_raw_data_tables, get_semantic_models
from semantic_modeling.interactive_modeling2 import expand_node, expand_edge
from semantic_modeling.karma.semantic_model import SemanticModel
from transformation.models.data_table import DataTable
from transformation.r2rml.r2rml import R2RML


def interactive_modeling(styper: SemanticTyper, sm: SemanticModel, tbl: DataTable):
    def print_stype(attr: str, ser_stype: str, expand: bool=False):
        node_id, type = ser_stype.split("---")
        if expand:
            node_id = expand_node(node_id)
            type = expand_edge(type)

        print(f"""-   _type_: SetSemanticType
    node_id: {node_id}
    domain: {node_id[:-1]}
    type: {type}
    input_attr_path: {attr}""")

    print("== SEMANTIC LABELING ==")
    styper.semantic_labeling_v2([sm], 4)

    for attr_path in tbl.schema.get_attr_paths():
        try:
            attr = sm.get_attr_by_label(attr_path)
        except KeyError:
            print("SKIP attr: %s because it's in SM" % (attr_path))
            continue

        attr_node = next(sm.graph.iter_nodes_by_label(attr_path.encode()))
        if attr_node.get_first_incoming_link().label != b"karma:dummy":
            print("SKIP attr: %s because it has been assigned stypes" % (attr_node.label.decode()))
        else:
            print("Attributes: %s. TOP 4 CHOICES:" % attr_path)
            for i, stype in enumerate(attr.semantic_types):
                print("[%s.] score = %.5f | %s --- %s" % (i + 1, stype.confidence_score, stype.domain, stype.type))
            choice = input("Your selection (1/2/3/4/s(kip)/manually enter it): ")
            if choice == 'skip' or choice == 's':
                pass
            elif choice.isdigit():
                idx = int(choice) - 1
                stype = attr.semantic_types[idx]
                print_stype(attr.label, f"{stype.domain}1---{stype.type}")
            else:
                # manually enter
                print_stype(attr.label, choice.strip(), True)


def gen_dummy_sm(sm: SemanticModel, tbl: DataTable):
    for attr_path in tbl.schema.get_attr_paths():
        try:
            attr = sm.get_attr_by_label(attr_path)
        except KeyError:
            print("""-   _type_: SetSemanticType
    node_id: crm:E12_Production1
    domain: crm:E12_Production
    type: karma:dummy
    input_attr_path: %s""" % attr_path)


if __name__ == '__main__':
    dataset = "museum_crm"
    ont = get_ontology(dataset)

    dataset_dir = Path(config.datasets[dataset].as_path())
    R2RML.load_python_scripts(Path(config.datasets[dataset].python_code.as_path()))

    # train the model first
    train_sms = get_semantic_models(dataset)[:-1]
    styper = SemanticTyper.get_instance(dataset, train_sms)

    # doing interactive modeling
    for tbl in get_raw_data_tables(dataset):
        if tbl.id in [sm.id for sm in train_sms]:
            continue

        print("Processing table:", tbl.id)
        print(tbl.head(10).to_string("double"))

        r2rml = R2RML.load_from_file(dataset_dir / "models-y2rml" / f"{tbl.id}-model.yml")
        sm = r2rml.apply_cmds(tbl)

        # gen_dummy_sm(sm, tbl)
        interactive_modeling(styper, sm, tbl)
        #
        new_tbl, sm = r2rml.apply_build(tbl.head(300))
        print(new_tbl.head(10).to_string("double"))
        sm.graph.render2pdf(dataset_dir / "models-viz" / f"{sm.id}.pdf")
        break