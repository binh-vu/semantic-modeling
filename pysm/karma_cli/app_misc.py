from typing import *


class NodeIdStr:

    def __init__(self, node_id_str: str) -> None:
        self.node_id_str = node_id_str

    def get_domain(self) -> str:
        return self.get_node_id()[:-1]

    def get_node_id(self) -> str:
        if self.node_id_str.endswith(" (add)"):
            return self.node_id_str.replace(" (add)", "")
        return self.node_id_str

    def is_new(self) -> bool:
        return self.node_id_str.endswith(" (add)")


class CliSemanticType:

    def __init__(self, node_id_str: NodeIdStr, predicate: str, score: float=0.0) -> None:
        self.node_id_str = node_id_str
        self.predicate = predicate
        self.score = score

    def get_domain(self) -> str:
        return self.node_id_str.get_domain()

    def get_node_id(self) -> str:
        return self.node_id_str.get_node_id()


class CliNode:

    def __init__(self, id: str, label: str, is_class_node: bool):
        self.id = id
        self.is_class_node = is_class_node
        self.label = label
        self.incoming_edges: List[CliEdge] = []
        self.outgoing_edges: List[CliEdge] = []


class CliEdge:

    def __init__(self, label: str, source_node: CliNode, target_node: CliNode):
        self.source_node = source_node
        self.target_node = target_node
        self.label = label


class CliGraph:

    def __init__(self):
        self.nodes: Dict[str, CliNode] = {}
        self.edges: List[CliEdge] = []

    def add_node(self, node_id: str):
        self.nodes[node_id] = CliNode(node_id, node_id[:-1], True)

    def add_data_node(self, attr: str):
        self.nodes[attr] = CliNode(attr, attr, False)

    def add_edge(self, label: str, source_node_id: str, target_node_id: str):
        edge = CliEdge(label, self.nodes[source_node_id], self.nodes[target_node_id])
        self.nodes[source_node_id].outgoing_edges.append(edge)
        self.nodes[source_node_id].outgoing_edges.append(edge)
        self.edges.append(edge)

    def remove_edge(self, edge: CliEdge):
        self.edges = [e for e in self.edges if e != edge]
        edge.source_node.outgoing_edges = [e for e in edge.source_node.outgoing_edges if e != edge]
        edge.target_node.incoming_edges = [e for e in edge.target_node.incoming_edges if e != edge]

    def get_node_by_id(self, node_id: str) -> CliNode:
        return self.nodes[node_id]

    def iter_nodes_by_label(self, lbl: str) -> Generator[CliNode, None, None]:
        for node in self.nodes.values():
            if node.label == lbl:
                yield node

    def get_class_node_ids(self) -> List[str]:
        results = []
        for node in self.nodes.values():
            if node.is_class_node:
                results.append(node.id)
        return results