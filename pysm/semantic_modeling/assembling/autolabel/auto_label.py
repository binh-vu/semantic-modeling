#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional

from data_structure import Graph
from semantic_modeling.assembling.autolabel.heuristic import preserved_structure_with_heuristic, get_gold_semantic_types
from semantic_modeling.assembling.autolabel.maxf1 import get_gold_triples, max_f1, max_f1_no_ambiguous
from semantic_modeling.assembling.autolabel.preserved_structure import preserved_structure


class AutoLabel:

    @staticmethod
    def auto_label_max_f1(gold_sm: Graph, pred_sm: Graph,
                          is_blurring_label: bool) -> Tuple[Dict[int, bool], Dict[int, Optional[int]], float]:
        gold_triples = get_gold_triples(gold_sm, is_blurring_label)
        return max_f1(gold_sm, pred_sm, is_blurring_label, gold_triples)

    @staticmethod
    def auto_label_max_f1_no_ambiguous(gold_sm: Graph, pred_sm: Graph, is_blurring_label: bool
                                      ) -> Tuple[Dict[int, bool], Dict[int, Optional[int]], float]:
        gold_triples = get_gold_triples(gold_sm, is_blurring_label)
        return max_f1_no_ambiguous(gold_sm, pred_sm, is_blurring_label, gold_triples)

    @staticmethod
    def auto_label_preserved_structure(gold_sm: Graph,
                                       pred_sm: Graph) -> Tuple[Dict[int, bool], Dict[int, Optional[int]]]:
        gold_triples = get_gold_triples(gold_sm, is_blurring_label=False)
        return preserved_structure(gold_sm, pred_sm, gold_triples)

    @staticmethod
    def auto_label_preserved_structure_heuristic_fix(
            gold_sm: Graph, pred_sm: Graph) -> Tuple[Dict[int, bool], Dict[int, Optional[int]]]:
        gold_triples = get_gold_triples(gold_sm, is_blurring_label=False)
        gold_stypes = get_gold_semantic_types(gold_sm)
        return preserved_structure_with_heuristic(gold_sm, pred_sm, gold_triples, gold_stypes)
