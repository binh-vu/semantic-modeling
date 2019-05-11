#!/usr/bin/python
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Dict

from semantic_labeling import create_semantic_typer
from semantic_labeling.sm_type_db import SemanticTypeDB
from semantic_labeling.typer import SemanticTyper
from semantic_modeling.assembling.ont_graph import get_ont_graph, PredicateType
from semantic_modeling.assembling.weak_models.cardinality_matrix import CardinalityFeatures
from semantic_modeling.assembling.weak_models.primary_key import PrimaryKey
from semantic_modeling.config import config
from semantic_modeling.data_io import get_short_train_name, get_semantic_models, get_ontology
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.settings import Settings
from semantic_modeling.utilities.serializable import serializeJSON


def serialize_ont_graph(dataset: str):
    def rdf_type_to_rust_str(rdf_type: PredicateType):
        if rdf_type == PredicateType.OWL_DATA_PROP:
            return "OwlDataProp"
        if rdf_type == PredicateType.OWL_OBJECT_PROP:
            return "OwlObjectProp"
        if rdf_type == PredicateType.OWL_ANNOTATION_PROP:
            return "OwlAnnotationProp"
        if rdf_type == PredicateType.RDF_PROP:
            return "RdfProp"
    ont = get_ontology(dataset)
    ont_graph = get_ont_graph(dataset)

    return {
        "predicates": [
            {
                "uri": ont.simplify_uri(predicate.uri),
                "domains": [ont.simplify_uri(uri) for uri in predicate.domains],
                "ranges": [ont.simplify_uri(uri) for uri in predicate.ranges],
                "rdf_type": rdf_type_to_rust_str(predicate.rdf_type),
                "is_rdf_type_reliable": predicate.is_rdf_type_reliable
            }
            for predicate in ont_graph.predicates
        ],
        "class_uris": {
            ont.simplify_uri(node.uri): {
                "uri": ont.simplify_uri(node.uri),
                "parents_uris": [ont.simplify_uri(uri) for uri in node.parents_uris],
                "children_uris": [ont.simplify_uri(uri) for uri in node.children_uris],
            }
            for node in ont_graph.iter_nodes()
        }
    }


def semantic_labeling(dataset: str, train_sms: List[SemanticModel], test_sms: List[SemanticModel]):
    if Settings.get_instance().semantic_labeling_simulate_testing:
        for sm in train_sms:
            custom_train_sms = [s for s in train_sms if s.id != sm.id]
            create_semantic_typer(dataset, custom_train_sms).semantic_labeling(
                custom_train_sms, [sm], top_n=Settings.get_instance().semantic_labeling_top_n_stypes, eval_train=False)
            SemanticTyper.instance = None # clear cache
            SemanticTypeDB.instance = None

        create_semantic_typer(dataset, train_sms).semantic_labeling(
            train_sms, test_sms, top_n=Settings.get_instance().semantic_labeling_top_n_stypes, eval_train=False)
    else:
        create_semantic_typer(dataset, train_sms).semantic_labeling(
            train_sms, test_sms, top_n=Settings.get_instance().semantic_labeling_top_n_stypes, eval_train=True)


def serialize_stype_assistant(dataset: str, sms: List[SemanticModel], train_sms: List[SemanticModel], test_sms: List[SemanticModel]):
    predicted_parent_stypes = SemanticTyper.get_instance(dataset, train_sms).semantic_labeling_parent(train_sms, test_sms, top_n=Settings.get_instance().semantic_labeling_top_n_stypes, eval_train=True)
    results = []

    for sm in sms:
        if sm.id not in predicted_parent_stypes:
            results.append({})
            continue

        all_parent_stypes = predicted_parent_stypes[sm.id]
        result = {}

        for attr_id, g_parent_stypes in all_parent_stypes.items():
            result[sm.graph.get_node_by_id(attr_id).label] = [{
                "stype": {
                    "domain": stype[0],
                    "type": stype[1],
                    "confidence_score": score
                },
                "parent_stypes": [{
                    "domain": parent_stype[0] if parent_stype is not None else "",
                    "type": parent_stype[1] if parent_stype is not None else "",
                    "confidence_score": parent_stype_score
                } for parent_stype, parent_stype_score in parent_stypes]
            } for stype, score, parent_stypes in g_parent_stypes]
        results.append(result)

    return results


def serialize_rust_input(dataset: str, workdir: str, train_sms: List[SemanticModel], test_sms: List[SemanticModel], foutput: Path):
    primary_key = PrimaryKey.get_instance(dataset, train_sms)

    sms = get_semantic_models(dataset)
    sm_index = {sm.id: i for i, sm in enumerate(sms)}
    train_sm_idxs = [sm_index[sm.id] for sm in train_sms]
    test_sm_idxs = [sm_index[sm.id] for sm in test_sms]

    predicted_parent_stypes = serialize_stype_assistant(dataset, sms, train_sms, test_sms)
    cardinality = CardinalityFeatures.get_instance(dataset)
    semantic_labeling(dataset, train_sms, test_sms)

    data = {
        "dataset": dataset,
        "workdir": str(workdir),
        "semantic_models": [sm.to_dict() for sm in sms],
        "predicted_parent_stypes": { "stype_details": predicted_parent_stypes },
        "train_sm_idxs": train_sm_idxs,
        "test_sm_idxs": test_sm_idxs,
        "feature_primary_keys": primary_key.to_dict(),
        "feature_cardinality_features": {
            sm_id: {
                "columns": matrix.columns,
                "matrix": matrix.matrix
            }
            for sm_id, matrix in cardinality.cardinality_matrices.items()
        },
        "ont_graph": serialize_ont_graph(dataset)
    }

    serializeJSON(data, foutput, indent=4)


if __name__ == '__main__':
    dataset = "museum_edm"
    source_models = get_semantic_models(dataset)
    train_sms = source_models[6:]
    test_sms = [sm for sm in source_models if sm not in train_sms]

    workdir = Path(config.fsys.debug.as_path()) / dataset / "main_experiments" / get_short_train_name(train_sms)
    workdir.mkdir(exist_ok=True, parents=True)
    (workdir / "rust").mkdir(exist_ok=True, parents=True)

    serialize_rust_input(dataset, workdir / "rust", train_sms, test_sms, workdir / "rust-input.json")