#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Generator, Tuple, List

from semantic_modeling.models.graph import Graph
from semantic_modeling.models.graph_link import WeightedGraphLink
from semantic_modeling.models.graph_node import GraphNode


class UnreachableNodeException(Exception):
    pass


class DijkstraSSSPIterator(object):

    def __init__(self, graph: Graph[GraphNode, WeightedGraphLink], source_id: str) -> None:
        self.graph: Graph[GraphNode, WeightedGraphLink] = graph
        self.distance: Dict[str, float] = {source_id: 0}
        self.prev_node: Dict[str, List[Tuple[str, str]]] = {source_id: None}
        self.source: GraphNode = graph.get_node_by_id(source_id)
        self.priority_queue = [self.source]

        for u in self.graph.iter_nodes_by_attrs():
            if u.id != source_id:
                self.distance[u.id] = float('inf')
                self.priority_queue.append(u)
        self.priority_queue.sort(key=lambda r: self.distance[r.id], reverse=True)

    def distance2nextnode(self):
        return self.distance[self.priority_queue[-1].id]

    def is_finish(self):
        return len(self.priority_queue) == 0 or self.distance[self.priority_queue[-1].id] == float('inf')

    def travel_next_node(self) -> Generator[GraphNode, None, None]:
        while len(self.priority_queue) > 0:
            u = self.priority_queue[-1]
            if self.distance[u.id] == float('inf'):
                # the remain nodes are unable to reach from source
                break

            self.priority_queue.pop()
            for l in u.iter_outgoing_links_by_attrs():
                v = l.get_dest_node()
                v_dist = self.distance[u.id] + l.weight
                if v_dist < self.distance[v.id]:
                    self.distance[v.id] = v_dist
                    self.prev_node[v.id] = [(u.id, l.id)]
                elif v_dist == self.distance[v.id]:
                    self.prev_node[v.id].append((u.id, l.id))
            self.priority_queue.sort(key=lambda r: self.distance[r.id], reverse=True)
            yield u

    def get_shortest_path(self, target_id: str) -> List[Tuple[str, str]]:
        if target_id not in self.prev_node:
            raise UnreachableNodeException("No path from source: `%s` to target: `%s`" % (self.source.id, target_id))

        path: List[Tuple[str, str]] = [(target_id, None)]
        while self.prev_node[target_id] is not None:
            path.append(self.prev_node[target_id][0])
            target_id = self.prev_node[target_id][0][0]

        return list(reversed(path))

    def get_all_shortest_paths(self, target_id: str) -> List[List[Tuple[str, str]]]:
        if target_id not in self.prev_node:
            raise UnreachableNodeException("No path from source: `%s` to target: `%s`" % (self.source.id, target_id))

        paths: List[List[Tuple[str, str]]] = [[(target_id, None)]]
        if self.prev_node[target_id] is not None:
            new_paths = []
            prev_paths = [
                (path, prev_n) for prev_n in self.prev_node[target_id] for path in self.get_all_shortest_paths(prev_n[0])
            ]
            for prev_n_path, prev_n in prev_paths:
                for path in paths:
                    new_path = prev_n_path[:-1] + [(prev_n_path[-1][0], prev_n[1])] + path
                    new_paths.append(new_path)

            return new_paths
        return paths

    def get_shortest_cost(self, target_id: str):
        return self.distance[target_id]
