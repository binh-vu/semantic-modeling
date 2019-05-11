#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Union, Optional, Any

from data_structure import Graph
from experiments.evaluation_metrics.semantic_modeling.pyeval import DependentGroups, Bijection, FindBestMapArgs, \
    iter_group_maps, eval_score, split_by_dependency, PairLabelGroup, prepare_args, DataNodeMode
from semantic_modeling.algorithm.combination import iter_index
from semantic_modeling.config import config


def find_all_best_map(dependent_group: DependentGroups, bijection: Bijection) -> List[Bijection]:
    terminate_index: int = len(dependent_group.pair_groups)
    # This code find the size of this array: sum([min(gold_group.size, pred_group.size) for gold_group, pred_group in dependent_group.groups])
    call_stack = [FindBestMapArgs(group_index=0, bijection=bijection)]
    n_called = 0

    best_map = []
    best_score = -1

    while True:
        call_args = call_stack.pop()
        n_called += 1
        if call_args.group_index == terminate_index:
            # it is terminated, calculate score
            score = eval_score(dependent_group, call_args.bijection)
            if score > best_score:
                best_score = score
                best_map = [call_args.bijection]
            elif score == best_score:
                best_map.append(call_args.bijection)
        else:
            pair_group = dependent_group.pair_groups[call_args.group_index]
            X, X_prime = pair_group.X, pair_group.X_prime
            for group_map in iter_group_maps(X, X_prime, call_args.bijection):
                bijection = Bijection.construct_from_mapping(group_map)

                call_stack.append(FindBestMapArgs(
                    group_index=call_args.group_index + 1,
                    bijection=call_args.bijection.extends(bijection)))

        if len(call_stack) == 0:
            break

    return best_map


def align_graph(gold_sm: Graph, pred_sm: Graph, data_node_mode: DataNodeMode):
    pair_groups: List[PairLabelGroup] = prepare_args(gold_sm, pred_sm, data_node_mode)

    mapping = []
    map_groups: List[PairLabelGroup] = []
    for pair in pair_groups:
        X, X_prime = pair.X, pair.X_prime

        if max(X.size, X_prime.size) == 1:
            x_prime = None if X_prime.size == 0 else X_prime.nodes[0].id
            x = None if X.size == 0 else X.nodes[0].id
            mapping.append((x_prime, x))
        else:
            map_groups.append(pair)

    partial_bijection = Bijection.construct_from_mapping(mapping)
    list_of_dependent_groups: List[DependentGroups] = split_by_dependency(map_groups, partial_bijection)

    n_permutations = sum([dependent_groups.get_n_permutations() for dependent_groups in list_of_dependent_groups])

    # TODO: remove debugging code or change to logging
    if n_permutations > 1000:
        print("Number of permutation is: %d" % n_permutations)

    if n_permutations > 1000000:
        gold_sm.render2img(config.fsys.debug.tmp.as_path() + "/gold.png")
        pred_sm.render2img(config.fsys.debug.tmp.as_path() + "/pred.png")
        for dependent_groups in list_of_dependent_groups:
            print(dependent_groups.pair_groups)
        assert False

    best_bijections = []
    for dependent_groups in list_of_dependent_groups:
        best_bijections.append(find_all_best_map(dependent_groups, partial_bijection))

    bijections = []
    for best_bijection_idxs in iter_index([len(x) for x in best_bijections]):
        bijection = Bijection()
        bijection.extends_(partial_bijection)
        for i, idx in enumerate(best_bijection_idxs):
            bijection.extends_(best_bijections[i][idx])
        bijections.append(bijection)

    all_groups = DependentGroups(pair_groups)
    TP = eval_score(all_groups, bijections[0])
    recall = TP / max(len(all_groups.X_triples), 1)
    precision = TP / max(len(all_groups.X_prime_triples), 1)
    if TP == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # remove a useless key which causes confusion
    for bijection in bijections:
        if None in bijection.prime2x:
            bijection.prime2x.pop(None)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        '_bijections': bijections
    }


