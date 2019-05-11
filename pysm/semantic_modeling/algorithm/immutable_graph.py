#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Dict, Iterable, List, TypeVar, Type, Generic, GenericMeta

from pyrsistent import PClass, field, pmap, PRecord, pvector, pmap_field, PClassMeta


class ImmutableLink(PRecord):

    id = field(type=str)
    label = field(type=str)
    source_id = field(type=str)
    target_id = field(type=str)

    @staticmethod
    def create(id: str, label: str, source_id: str, target_id: str) -> 'ImmutableLink':
        return ImmutableLink(id=id, label=label, source_id=source_id, target_id=target_id)


class ImmutableNode(PRecord):
    id: str = field(type=str)
    label: str = field(type=str)
    incoming_links: List[ImmutableLink] = field()
    outgoing_links: List[ImmutableLink] = field()

    @classmethod
    def create(cls, id: str, label: str, incoming_links: List[ImmutableLink]=None, outgoing_links: List[ImmutableLink]=None) -> 'ImmutableNode':
        if incoming_links is None:
            incoming_links = []
        if outgoing_links is None:
            outgoing_links = []

        return cls(id=id, label=label, incoming_links=pvector(incoming_links), outgoing_links=pvector(outgoing_links))

    def add_incoming_link(self, link: ImmutableLink) -> 'ImmutableNode':
        return self.set('incoming_links', self.incoming_links.append(link))

    def add_incoming_links(self, links: List[ImmutableLink]) -> 'ImmutableNode':
        return self.set('incoming_links', self.incoming_links.extend(links))

    def add_outgoing_link(self, link: ImmutableLink) -> 'ImmutableNode':
        return self.set('outgoing_links', self.outgoing_links.append(link))

    def add_outgoing_links(self, links: List[ImmutableLink]) -> 'ImmutableNode':
        return self.set('outgoing_links', self.outgoing_links.extend(links))

    def remove_incoming_link(self, link: ImmutableLink) -> 'ImmutableNode':
        return self.set('incoming_links', self.incoming_links.remove(link))

    def remove_outgoing_link(self, link: ImmutableLink) -> 'ImmutableNode':
        return self.set('outgoing_links', self.outgoing_links.remove(link))


ImmutableNodeType = TypeVar('ImmutableNodeType', covariant=True, bound=ImmutableNode)
ImmutableLinkType = TypeVar('ImmutableLinkType', covariant=True, bound=ImmutableLink)


class ImmutableGraphMeta(GenericMeta, PClassMeta):
    pass


class ImmutableGraph(PClass, Generic[ImmutableNodeType, ImmutableLinkType], metaclass=ImmutableGraphMeta):

    nodes: Dict[str, ImmutableNodeType] = pmap_field(key_type=str, value_type=ImmutableNode)
    links: Dict[str, ImmutableLinkType] = pmap_field(key_type=str, value_type=ImmutableLink)

    @classmethod
    def create(cls, n_nodes: int, n_links: int) -> 'ImmutableGraph':
        nodes = pmap({}, n_nodes)
        links = pmap({}, n_links)
        return cls(nodes=nodes, links=links)

    def generate_node_id(self, prefix: str='', offset: int=0) -> str:
        count = len(self.nodes) + offset
        if prefix != '':
            prefix = '-' + prefix
        id = 'N%s%03d' % (prefix, count)
        assert id not in self.nodes, 'The generated id: %s is already defined' % id
        return id

    def generate_link_id(self, prefix: str='', offset: int=0) -> str:
        count = len(self.links) + offset
        if prefix != '':
            prefix = '-' + prefix
        id = 'L%s%03d' % (prefix, count)
        assert id not in self.links, 'The generated id: %s is already defined' % id
        return id

    def evolver(self):
        return ImmutableGraphEvolver(self.__class__, self.nodes.evolver(), self.links.evolver())

    def batch_update(self, nodes: List[ImmutableNodeType], links: List[ImmutableLinkType]) -> 'ImmutableGraph[ImmutableNodeType, ImmutableLinkType]':
        outgoing_links = defaultdict(lambda: [])
        incoming_links = defaultdict(lambda: [])

        for link in links:
            outgoing_links[link.source_id].append(link)
            incoming_links[link.target_id].append(link)

        updated_nodes: Dict[str, ImmutableNodeType] = {n.id: n for n in nodes}
        for node_id in outgoing_links:
            if node_id not in updated_nodes:
                updated_nodes[node_id] = self.nodes[node_id]
            updated_nodes[node_id] = updated_nodes[node_id].add_outgoing_links(outgoing_links[node_id])

        for node_id in incoming_links:
            if node_id not in updated_nodes:
                updated_nodes[node_id] = self.nodes[node_id]
            updated_nodes[node_id] = updated_nodes[node_id].add_incoming_links(incoming_links[node_id])

        return self.set(
            nodes=self.nodes.update(updated_nodes),
            links=self.links.update({l.id: l for l in links}))

    def save_node(self, node: ImmutableNodeType) -> 'ImmutableGraph[ImmutableNodeType, ImmutableLinkType]':
        return self.set('nodes', self.nodes.set(node.id, node))

    def save_link(self, link: ImmutableLinkType) -> 'ImmutableGraph[ImmutableNodeType, ImmutableLinkType]':
        source_node = self.nodes.get(link.source_id)
        target_node = self.nodes.get(link.target_id)

        source_node = source_node.add_outgoing_link(link)
        target_node = target_node.add_incoming_link(link)

        return self.set(
            nodes=self.nodes.update({link.source_id: source_node, link.target_id: target_node}),
            links=self.links.set(link.id, link)
        )

    def remove_node(self, node: ImmutableNodeType) -> 'ImmutableGraph[ImmutableNodeType, ImmutableLinkType]':
        G = self
        for incoming_link in node.incoming_links:
            G = G.remove_link(incoming_link)
        for outgoing_link in node.outgoing_links:
            G = G.remove_link(outgoing_link)

        nodes = G.nodes.remove(node.id)
        return self.__class__(nodes=nodes, links=G.links)

    def remove_link(self, link: ImmutableLinkType) -> 'ImmutableGraph[ImmutableNodeType, ImmutableLinkType]':
        links = self.links.remove(link.id)
        nodes = self.nodes.update({
            link.source_id: self.nodes.get(link.source_id).remove_outgoing_link(link),
            link.target_id: self.nodes.get(link.target_id).remove_incoming_link(link)
        })

        return self.__class__(nodes=nodes, links=links)

    def has_node_with_id(self, id: str) -> bool:
        return id in self.nodes

    def has_link_with_id(self, id: str) -> bool:
        return id in self.links

    def get_node_by_id(self, id: str) -> ImmutableNodeType:
        return self.nodes.get(id)

    def get_link_by_id(self, id: str) -> ImmutableLinkType:
        return self.links.get(id)

    def get_source_node(self, link: ImmutableLinkType) -> ImmutableNodeType:
        return self.nodes.get(link.source_id)

    def get_target_node(self, link: ImmutableLinkType) -> ImmutableNodeType:
        return self.nodes.get(link.target_id)

    def iter_root_nodes(self) -> Iterable[ImmutableNodeType]:
        return (n for n in self.nodes.values() if len(n.incoming_links) == 0)

    def get_root_nodes(self) -> List[ImmutableNodeType]:
        return [n for n in self.nodes.values() if len(n.incoming_links) == 0]

    def consistent_tree_hashing(self) -> str:
        """Create a hashing for this graph, but it only valid if this graph is a "tree" (may have multiple roots)"""
        def tree_hashing(graph: ImmutableGraph, node: ImmutableNodeType) -> str:
            children_texts = []
            for link in node.outgoing_links:
                child = graph.get_target_node(link)
                children_texts.append(tree_hashing(graph, child))
            return "(%s:%s)" % (node.label, ",".join(sorted(children_texts)))

        roots = []
        for r in self.iter_root_nodes():
            roots.append(tree_hashing(self, r))
        return ",".join(sorted(roots))


T = TypeVar('T', bound=ImmutableGraph, covariant=True)


class ImmutableGraphEvolver(object):

    def __init__(self, cls: Type[T], nodes: Dict[str, ImmutableNode], links: Dict[str, ImmutableLink]) -> None:
        self.nodes = nodes
        self.links = links
        self.cls: Type[T] = cls

    def generate_node_id(self, prefix: str='', offset: int=0) -> str:
        count = len(self.nodes) + offset
        if prefix != '':
            prefix = '-' + prefix
        id = 'N%s%03d' % (prefix, count)
        assert id not in self.nodes, 'The generated id: %s is already defined' % id
        return id

    def generate_link_id(self, prefix: str='', offset: int=0) -> str:
        count = len(self.links) + offset
        if prefix != '':
            prefix = '-' + prefix
        id = 'L%s%03d' % (prefix, count)
        assert id not in self.links, 'The generated id: %s is already defined' % id
        return id

    def save_node(self, node: ImmutableNode):
        self.nodes[node.id] = node

    def save_link(self, link: ImmutableLink):
        self.links[link.id] = link
        self.nodes[link.source_id] = self.nodes[link.source_id].add_outgoing_link(link)
        self.nodes[link.target_id] = self.nodes[link.target_id].add_incoming_link(link)

    def has_node_with_id(self, id: str) -> bool:
        return id in self.nodes

    def has_link_with_id(self, id: str) -> bool:
        return id in self.links

    def get_node_by_id(self, id: str) -> ImmutableNode:
        return self.nodes[id]

    def get_link_by_id(self, id: str) -> ImmutableLink:
        return self.links[id]

    def get_source_node(self, link: ImmutableLink) -> ImmutableNode:
        return self.nodes[link.source_id]

    def get_target_node(self, link: ImmutableLink) -> ImmutableNode:
        return self.nodes[link.target_id]

    def iter_root_nodes(self) -> Iterable[ImmutableNode]:
        return (n for n in self.nodes.values() if len(n.incoming_links) == 0)

    def get_root_nodes(self) -> List[ImmutableNode]:
        return [n for n in self.nodes.values() if len(n.incoming_links) == 0]

    def persistent(self) -> T:
        return self.cls(nodes=self.nodes.persistent(), links=self.links.persistent())


if __name__ == '__main__':
    class InheritImmutableNode(ImmutableNode):
        type: str = field(type=str)

        @classmethod
        def create(cls, id: str, label: str, type: str, incoming_links: List[ImmutableLink] = None,
                   outgoing_links: List[ImmutableLink] = None) -> 'ImmutableNode':
            return super(InheritImmutableNode, cls).create(id, label, incoming_links, outgoing_links).set('type', type)


    G = ImmutableGraph()
    g = G.evolver()

    n0 = ImmutableNode.create('N000', 'Person0')
    n1 = ImmutableNode.create('N001', 'Person1')
    n2 = ImmutableNode.create('N002', 'Person2')
    n3 = ImmutableNode.create('N003', 'Person3')
    n4 = InheritImmutableNode.create('N004', 'Person4', 'DataNode')

    g.save_node(n0)
    g.save_node(n1)
    g.save_node(n2)
    g.save_node(n3)
    g.save_node(n4)

    G1 = g.persistent()
    print(G1)
