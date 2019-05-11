#!/usr/bin/python
# -*- coding: utf-8 -*-
from itertools import chain
from pathlib import Path

from semantic_labeling import create_semantic_typer
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.config import config
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.link_prediction.int_graph import IntGraph, add_known_models, IntGraphNode, create_psl_int_graph
from semantic_modeling.link_prediction.oracle_prediction import oracle_link_prediction
from semantic_modeling.utilities.serializable import serializeCSV


def extract_predicates(g: IntGraph, sm: SemanticModel, statistic: Statistic):
    true_links = oracle_link_prediction(int_graph, sm)

    semantic_label_predicates = []
    triple_predicates = []
    class_node_lbl_predicates = {}
    link_predicates = []

    for attr in sm.attrs:
        for st in attr.semantic_types:
            n: IntGraphNode
            for n in int_graph.iter_nodes_by_label(st.domain.encode("utf-8")):
                semantic_label_predicates.append((n.readable_id, st.type, attr.label, st.confidence_score))
                class_node_lbl_predicates[n.readable_id] = (n.readable_id, n.label.decode('utf-8'))

    for s, p, o, score in semantic_label_predicates:
        spo = (s, p.encode('utf-8'), o)
        triple_predicates.append((s, p, o, spo in true_links))

    for e in g.iter_links():
        source = e.get_source_node()
        target = e.get_target_node()
        spo = (source.readable_id, e.label, target.readable_id)
        triple_predicates.append((spo[0], e.label.decode('utf-8'), spo[2], spo in true_links))
        class_node_lbl_predicates[source.readable_id] = (source.readable_id, source.label.decode('utf-8'))
        class_node_lbl_predicates[target.readable_id] = (target.readable_id, target.label.decode('utf-8'))
        link_predicates.append((spo[0], e.label.decode('utf-8'), spo[2], statistic.p_l_given_so(source.label, e.label, target.label, 0.3)))

    return semantic_label_predicates, triple_predicates, class_node_lbl_predicates.values(), link_predicates


def norm_psl_constant(s):
    return s.replace("-", "_").replace(":", "_").replace(" ", "").replace("(", "-").replace(")", "-")


def norm_psl_constants(array: list):
    if isinstance(array[0], (list, tuple)):
        for i in range(len(array)):
            array[i] = [norm_psl_constant(s) if isinstance(s, str) else s for s in array[i]]
    else:
        for i in range(len(array)):
            array[i] = norm_psl_constant(array[i]) if isinstance(array[i], str) else array[i]
    return array


if __name__ == '__main__':
    dataset = "museum_edm"
    semantic_models = get_semantic_models(dataset)
    train_sms = semantic_models[6:]
    test_sms = semantic_models[:2]

    statistic = Statistic.get_instance(train_sms)

    workdir = Path(config.fsys.debug.as_path()) / dataset / "psl"
    (workdir / "data").mkdir(exist_ok=True, parents=True)

    # int_graph = IntGraph(True, True, True, 100, 100)
    # add_known_models(int_graph, semantic_models[:train_size])
    int_graph = create_psl_int_graph(train_sms)
    int_graph.render2img(workdir / "int_graph.png")

    files = {
        "SemanticLabelObsFile": workdir / "data" / "semantic_lbl_obs.txt",
        "IsClassObsFile": workdir / "data" / "is_class_obs.txt",
        "LinkObsFile": workdir / "data" / "link_obs.txt",
        "DataNodeObsFile": workdir / "data" / "data_node_obs.txt",
        "OpenCooccurrence": workdir / "data" / "open_cooccurrence_obs.txt",
        "TripleObsFile": workdir / "data" / "triple_obs.txt",
        "TripleTargetFile": workdir / "data" / "triple_target.txt",
        "TripleTruthFile": workdir / "data" / "triple_truth.txt",
    }

    with open(str(Path(__file__).parent / "semantic_mapping.data.template"), "r") as f, \
        open(workdir / "semantic_mapping.data", "w") as g:
            g.write(f.read().format(**files))

    # MAKE PSL RULES
    with open(str(workdir / "semantic_mapping.psl"), "w") as f:
        psl_rules = [
            "2: !Triple(S, P, D)^2",
            "Triple(+S, +P, D) <= 1 .",  # each node should have only one incoming links
            "1: Triple(S, P, D) && Triple(S2, P2, D2) && Link(S2, P3, S) && S != D2 && S != S2 -> Triple(S2, P3, S)",
            "1: SemanticLabel(S, P, D) -> Triple(S, P, D)",
            "1: Triple(S, P, D) && SemanticLabel(S, P, D2) && D != D2 -> !Triple(S, P, D2)",
            "1: Triple(S, P, D) && Link(S, P, D2) && D != D2 -> !Triple(S, P, D2)",
            "1: Cooccurrence(S, P, P2) & Triple(S, P, D) && SemanticLabel(S, P2, D2) -> Triple(S, P2, D2)"
        ]
        for rule in psl_rules:
            f.write(rule + "\n")

        # f.write("\n// semantic labeling rules per semantic-types\n\n")
        #
        # _tmpvar = set()
        # for sm in train_sms:
        #     for n in sm.graph.iter_class_nodes():
        #         for e in n.iter_outgoing_links():
        #             _tmpvar.add(f"IsClass(S, '{n.label.decode('utf-8')}') && P == '{e.label.decode('utf-8')}'")
        #
        # for additional_rule in _tmpvar:
        #     additional_rule = norm_psl_constant(additional_rule)
        #     f.write(f"1: SemanticLabel(S, P, D) && {additional_rule} -> Triple(S, P, D)\n")
        #
        # f.write("\n// duplication predicates\n\n")
        # for additional_rule in _tmpvar:
        #     additional_rule = norm_psl_constant(additional_rule)
        #     f.write(f"1: Triple(S, P, D) && SemanticLabel(S, P, D2) && D != D2 && {additional_rule} -> !Triple(S, P, D2)\n")
        #     f.write(f"1: Triple(S, P, D) && Link(S, P, D2) && D != D2 && {additional_rule} -> !Triple(S, P, D2)\n")
        #
        # f.write("\n// co-occurrence rules\n\n")
        # _tmpvar = set()
        # for sm in train_sms:
        #     for n in sm.graph.iter_class_nodes():
        #         for e in n.iter_outgoing_links():
        #             for e2 in n.iter_outgoing_links():
        #                 if e.label == e2.label:
        #                     continue
        #                 _tmpvar.add(f"IsClass(S, '{n.label.decode('utf-8')}') && P == '{e.label.decode('utf-8')}' && P2 == '{e2.label.decode('utf-8')}'")
        #
        # for additional_rule in _tmpvar:
        #     additional_rule = norm_psl_constant(additional_rule)
        #     f.write(f"1: Triple(S, P, D) && SemanticLabel(S, P2, D2) && {additional_rule} -> Triple(S, P2, D2)\n")

    semantic_typer = create_semantic_typer(dataset, train_sms)
    semantic_typer.semantic_labeling(train_sms, test_sms, 4, True)

    # make semantic labeling
    predicates = {
        "SemanticLabel": [],
        "IsClass": [],
        "DataNode": [],
        "Link": [],
        "Triple": {
            "obs": [],
            "target": [],
            "truth": []
        }
    }

    atoms = set()
    seen_triples = set()

    for i, sm in enumerate(chain(train_sms, test_sms)):
        sid = sm.id[:3]
        st_predicates, triple_obs, class_node_lbls, link_obs = extract_predicates(int_graph, sm, statistic)
        for attr in sm.attrs:
            predicates['DataNode'].append((f"{sid}::{attr.label}", 1))
            atoms.add(f"{sid}::{attr.label}")

        for node_id, lbl in class_node_lbls:
            predicates['IsClass'].append((f"{sid}::{node_id}", lbl, 1))
            atoms.add(f"{sid}::{node_id}")

        for s, p, o, score in st_predicates:
            predicates['SemanticLabel'].append((f"{sid}::{s}", p, f"{sid}::{o}", score))
            atoms.add(f"{sid}::{s}")
            atoms.add(p)
            atoms.add(f"{sid}::{o}")

        for s, p, o, p_link in link_obs:
            predicates['Link'].append((f"{sid}::{s}", p, f"{sid}::{o}", p_link))
            atoms.add(f"{sid}::{s}")
            atoms.add(p)
            atoms.add(f"{sid}::{o}")

        if i < len(train_sms):
            for s, p, o, label in triple_obs:
                predicates['Triple']['obs'].append((f"{sid}::{s}", p, f"{sid}::{o}", int(label)))
                seen_triples.add((f"{sid}::{s}", p, f"{sid}::{o}"))
        else:
            for s, p, o, label in triple_obs:
                predicates['Triple']['target'].append((f"{sid}::{s}", p, f"{sid}::{o}", 0))
                predicates['Triple']['truth'].append((f"{sid}::{s}", p, f"{sid}::{o}", int(label)))
                seen_triples.add((f"{sid}::{s}", p, f"{sid}::{o}"))

    for s in atoms:
        for p in atoms:
            for o in atoms:
                if (s, p, o) not in seen_triples:
                    predicates['Triple']['obs'].append((s, p, o, 0))

    serializeCSV(norm_psl_constants(predicates['SemanticLabel']), files['SemanticLabelObsFile'], delimiter="\t")
    serializeCSV(norm_psl_constants(predicates['IsClass']), files['IsClassObsFile'], delimiter="\t")
    serializeCSV(norm_psl_constants(predicates['DataNode']), files['DataNodeObsFile'], delimiter="\t")
    serializeCSV(norm_psl_constants(predicates['Link']), files['LinkObsFile'], delimiter="\t")
    serializeCSV(norm_psl_constants(predicates['Triple']['obs']), files['TripleObsFile'], delimiter='\t')
    serializeCSV(norm_psl_constants(predicates['Triple']['target']), files['TripleTargetFile'], delimiter='\t')
    serializeCSV(norm_psl_constants(predicates['Triple']['truth']), files['TripleTruthFile'], delimiter='\t')