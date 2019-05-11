#!/usr/bin/python
# -*- coding: utf-8 -*-
from data_structure import Graph, GraphLink, GraphNode


class IntegrationPoint(object):
    is_incoming_link: bool
    link_id: bool
    source_id: bool
    target_id: bool

    def __init__(self, is_incoming_link: bool, link_id: bool, source_id: bool, target_id: bool) -> None: ...

    def to_dict(self) -> dict: ...


class MergeGraph(Graph[GraphNode, GraphLink]):

    def __init__(self, index_node_type: bool = False, index_node_label: bool = False, index_link_label: bool = False,
                 estimated_n_nodes: int = 0, estimated_n_links: int = 0, name: bytes = b"graph", g_part_a: Graph = None,
                 g_part_b: Graph = None, g_part_merge: Graph = None, point_a: IntegrationPoint = None,
                 point_b: IntegrationPoint = None) -> None: ...

    @staticmethod
    def create(g_part_a: Graph, g_part_b: Graph, g_part_merge: Graph, point_a: IntegrationPoint, point_b: IntegrationPoint) -> MergeGraph: ...

    def proceed_merging(self) -> Graph: ...
