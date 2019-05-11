#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional

from data_structure import Graph
from semantic_modeling.methods.assembling.cshare.merge_graph import IntegrationPoint


class MergePlan:

    int_tree: Graph
    int_a: IntegrationPoint
    int_b: IntegrationPoint

    def __init__(self, int_tree: Graph, int_a: IntegrationPoint, int_b: IntegrationPoint) -> None: ...


def make_merge_plans(treeA: Graph, treeB: Graph, treeAsearch: Graph, treeBsearch: Graph) -> List[MergePlan]: ...

def py_make_plan4case23(treeA: Graph, treeB: Graph, treeAsearch: Graph, treeBsearch: Graph) -> List[MergePlan]: ...

def py_make_plan4case1(treeA: Graph, treeB: Graph, treeAsearch: Graph, treeBsearch: Graph) -> List[MergePlan]: ...