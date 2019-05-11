#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
from enum import IntEnum
from itertools import permutations, chain
from typing import Dict, Tuple, List, Set, Optional, Callable, Generator

from pyrsistent import pvector, PVector

from data_structure import Graph
from semantic_modeling.config import config
from semantic_modeling.data_io import get_karma_models
from semantic_modeling.utilities.serializable import deserialize
"""
Convention: x' and x are nodes in predicted model and gold model, respectively.
"""

class PermutationExploding(Exception):
    pass


class Node(object):
    CLASS_NODE = 0
    DATA_NODE = 1

    def __init__(self, id: int, type: int, label: bytes) -> None:
        self.id: int = id
        self.label: bytes = label
        self.type: int = type
        self.incoming_links: List[Link] = []
        self.outgoing_links: List[Link] = []

    @staticmethod
    def is_data_node(self: 'Node'):
        return self.type == Node.DATA_NODE

    @staticmethod
    def add_incoming_link(self: 'Node', link: 'Link'):
        self.incoming_links.append(link)
        link.target = self

    @staticmethod
    def add_outgoing_link(self: 'Node', link: 'Link'):
        self.outgoing_links.append(link)
        link.source = self

    def __str__(self):
        return "Node(id=%s, label=%s)" % (self.id, self.label)


class Link(object):
    def __init__(self, id: int, label: bytes, source_id: int, target_id: int) -> None:
        self.id: int = id
        self.label: bytes = label
        self.target_id: int = target_id
        self.source_id: int = source_id
        self.source: Node = None
        self.target: Node = None


class LabelGroup(object):
    """Represent a group of nodes that have same label"""

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes: List[Node] = nodes
        self.node_triples: Set[Tuple[int, bytes, int, int]] = {
            (link.source_id, link.label, link.target_id
             if link.target.type == Node.CLASS_NODE else link.target.label, link.target.type)
            for node in self.nodes for link in chain(node.incoming_links, node.outgoing_links)
        }
        self.size: int = len(nodes)

    def __repr__(self):
        return "(#nodes=%d)" % (len(self.nodes))

    # noinspection PyUnusedLocal
    @staticmethod
    def group_by_structures(group: 'LabelGroup', pred_group: 'LabelGroup'):
        """
            A structure of a node is defined by its links, or we can treat it as a set of triple.
            Unbounded nodes should be assumed to be different, therefore a node have unbounded nodes will have
            it own structure group.
            We need not consider triple that are impossible to map to node in pred_group. This trick will improve
            the performance.
        """
        # TODO: implement it
        return [StructureGroup([n]) for n in group.nodes]


class StructureGroup(object):
    """Represent a group of nodes that have same structure"""

    def __init__(self, nodes: List[Node]) -> None:
        self.nodes: List[Node] = nodes
        self.size: int = len(nodes)


class PairLabelGroup(object):
    def __init__(self, label: bytes, X: LabelGroup, X_prime: LabelGroup) -> None:
        self.label: bytes = label
        self.X: LabelGroup = X
        self.X_prime: LabelGroup = X_prime

    def __repr__(self):
        return "(label=%s, X=%s, X_prime=%s)" % (self.label.decode("utf-8"), self.X, self.X_prime)


class DependentGroups(object):
    """Represent a list of groups of nodes that are dependent on each other"""

    def __init__(self, pair_groups: List[PairLabelGroup]):
        self.pair_groups: List[PairLabelGroup] = pair_groups
        self.X_triples: Set[Tuple[int, bytes, int, int]] = pair_groups[0].X.node_triples
        self.X_prime_triples: Set[Tuple[int, bytes, int, int]] = pair_groups[0].X_prime.node_triples

        for pair in pair_groups[1:]:
            self.X_triples = self.X_triples.union(pair.X.node_triples)
            self.X_prime_triples = self.X_prime_triples.union(pair.X_prime.node_triples)

    def get_n_permutations(self):
        n_permutation = 1
        for pair_group in self.pair_groups:
            n = max(pair_group.X.size, pair_group.X_prime.size)
            m = min(pair_group.X.size, pair_group.X_prime.size)
            n_permutation *= math.factorial(n) / math.factorial(n - m)

        return n_permutation


class Bijection(object):
    """A bijection defined a one-one mapping from x' => x"""

    def __init__(self) -> None:
        # a map from x' => x (pred_sm to gold_sm)
        self.prime2x: Dict[int, int] = {}
        # map from x => x'
        self.x2prime: Dict[int, int] = {}

    @staticmethod
    def construct_from_mapping(mapping: List[Tuple[Optional[int], Optional[int]]]) -> 'Bijection':
        """
        :param mapping: a list of map from x' => x
        """
        self = Bijection()
        self.prime2x: Dict[int, int] = {x_prime: x for x_prime, x in mapping}
        self.x2prime: Dict[int, int] = {x: x_prime for x_prime, x in mapping}
        return self

    def extends(self, bijection: 'Bijection') -> 'Bijection':
        another = Bijection()
        another.prime2x = dict(self.prime2x)
        another.prime2x.update(bijection.prime2x)
        another.x2prime = dict(self.x2prime)
        another.x2prime.update(bijection.x2prime)
        return another

    def extends_(self, bijection: 'Bijection') -> None:
        self.prime2x.update(bijection.prime2x)
        self.x2prime.update(bijection.x2prime)

    def is_gold_node_bounded(self, node_id: int) -> bool:
        return node_id in self.x2prime

    def is_pred_node_bounded(self, node_id: int) -> bool:
        return node_id in self.prime2x


class IterGroupMapsUsingGroupingArgs(object):
    def __init__(self, node_index: int, bijection: PVector, G_sizes) -> None:
        self.node_index: int = node_index
        self.bijection: PVector = bijection
        self.G_sizes = G_sizes


class IterGroupMapsGeneralApproachArgs(object):
    def __init__(self, node_index: int, bijection: PVector) -> None:
        self.node_index: int = node_index
        self.bijection: PVector = bijection


class FindBestMapArgs(object):
    def __init__(self, group_index: int, bijection: Bijection) -> None:
        self.group_index: int = group_index
        self.bijection = bijection


def eval_score(dependent_groups: DependentGroups, bijection: Bijection) -> float:
    X_triples = dependent_groups.X_triples
    X_prime_triples = set()

    for triple in dependent_groups.X_prime_triples:
        mapped_source: str = triple[0] if triple[0] not in bijection.prime2x else bijection.prime2x[triple[0]]
        mapped_target: str = triple[2] if triple[3] == Node.DATA_NODE else (bijection.prime2x[triple[2]] if triple[2] in bijection.prime2x else triple[2])

        new_triple = (mapped_source, triple[1], mapped_target, triple[3])

        X_prime_triples.add(new_triple)

    return float(len(X_triples.intersection(X_prime_triples)))


def find_best_map(dependent_group: DependentGroups, bijection: Bijection) -> Bijection:
    terminate_index: int = len(dependent_group.pair_groups)
    # This code find the size of this array: sum([min(gold_group.size, pred_group.size) for gold_group, pred_group in dependent_group.groups])
    call_stack = [FindBestMapArgs(group_index=0, bijection=bijection)]
    n_called = 0

    best_map = None
    best_score = -1

    while True:
        call_args = call_stack.pop()
        n_called += 1
        if call_args.group_index == terminate_index:
            # it is terminated, calculate score
            score = eval_score(dependent_group, call_args.bijection)
            if score > best_score:
                best_score = score
                best_map = call_args.bijection
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


def get_unbounded_nodes(X: LabelGroup, is_bounded_func: Callable[[int], bool]) -> List[Node]:
    """Get nodes of a label group which have not been bounded by a bijection"""
    unbounded_nodes = []

    for x in X.nodes:
        for link in x.incoming_links:
            if not is_bounded_func(link.source_id):
                unbounded_nodes.append(link.source)

        for link in x.outgoing_links:
            if link.target.type == Node.CLASS_NODE and not is_bounded_func(link.target_id):
                unbounded_nodes.append(link.target)

    return unbounded_nodes


def get_common_unbounded_nodes(X: LabelGroup, X_prime: LabelGroup, bijection: Bijection) -> Set[bytes]:
    """Finding unbounded nodes in X and X_prime that have same labels"""
    unbounded_X = get_unbounded_nodes(X, bijection.is_gold_node_bounded)
    unbounded_X_prime = get_unbounded_nodes(X_prime, bijection.is_pred_node_bounded)

    labeled_unbounded_X = {}
    labeled_unbounded_X_prime = {}
    for x in unbounded_X:
        if x.label not in labeled_unbounded_X:
            labeled_unbounded_X[x.label] = []
        labeled_unbounded_X[x.label].append(x)

    for x in unbounded_X_prime:
        if x.label not in labeled_unbounded_X_prime:
            labeled_unbounded_X_prime[x.label] = []
        labeled_unbounded_X_prime[x.label].append(x)

    common_unbounded_nodes = set(labeled_unbounded_X.keys()).intersection(labeled_unbounded_X_prime.keys())
    return common_unbounded_nodes


def group_dependent_elements(dependency_map: List[List[int]]) -> List[int]:
    # algorithm to merge the dependencies
    # input:
    #   - dependency_map: [<element_index, ...>, ...] list of dependencies where element at ith position is list of index of elements
    #           that element at ith position depends upon.
    # output:
    #   - dependency_groups: [<group_id>, ...] list of group id, where element at ith position is group id that element belongs to
    dependency_groups: List[int] = [-1 for _ in range(len(dependency_map))]
    invert_dependency_groups = {}

    for i, g in enumerate(dependency_map):
        dependent_elements = g + [i]
        groups = {dependency_groups[j] for j in dependent_elements}
        valid_groups = groups.difference([-1])
        if len(valid_groups) == 0:
            group_id = len(invert_dependency_groups)
            invert_dependency_groups[group_id] = set()
        else:
            group_id = next(iter(valid_groups))

        if -1 in groups:
            # map unbounded elements to group has group_id
            for j in dependent_elements:
                if dependency_groups[j] == -1:
                    dependency_groups[j] = group_id
                    invert_dependency_groups[group_id].add(j)

        for another_group_id in valid_groups.difference([group_id]):
            for j in invert_dependency_groups[another_group_id]:
                dependency_groups[j] = group_id
                invert_dependency_groups[group_id].add(j)

    return dependency_groups


def split_by_dependency(map_groups: List[PairLabelGroup], bijection: Bijection) -> List[DependentGroups]:
    """This method takes a list of groups (X, X') and group them based on their dependencies.
    D = {D1, D2, …} s.t for all Di, Dj, (Xi, Xi') in Di, (Xj, Xj’) in Dj, they are independent

    Two groups of nodes are dependent when at least one unbounded nodes in a group is a label of other group.
    For example, "actor_appellation" has link to "type", so group "actor_appellation" depends on group "type"
    """
    group_label2idx = {map_group.label: i for i, map_group in enumerate(map_groups)}

    # build group dependencies
    dependency_map = [[] for _ in range(len(map_groups))]
    for i, map_group in enumerate(map_groups):
        X, X_prime = map_group.X, map_group.X_prime
        common_labels = get_common_unbounded_nodes(X, X_prime, bijection)

        for common_label in common_labels:
            group_id = group_label2idx[common_label]
            dependency_map[i].append(group_id)

    dependency_groups = group_dependent_elements(dependency_map)
    dependency_pair_groups: Dict[int, List[PairLabelGroup]] = {}
    dependency_map_groups: List[DependentGroups] = []

    for i, map_group in enumerate(map_groups):
        if dependency_groups[i] not in dependency_pair_groups:
            dependency_pair_groups[dependency_groups[i]] = []
        dependency_pair_groups[dependency_groups[i]].append(map_group)

    for pair_groups in dependency_pair_groups.values():
        dependency_map_groups.append(DependentGroups(pair_groups))

    return dependency_map_groups


# noinspection PyUnusedLocal
def iter_group_maps(X: LabelGroup, X_prime: LabelGroup,
                    bijection: Bijection) -> Generator[List[Tuple[int, int]], None, None]:
    if X.size < X_prime.size:
        return iter_group_maps_general_approach(X, X_prime)
    else:
        G = LabelGroup.group_by_structures(X, X_prime)
        return iter_group_maps_using_grouping(X_prime, G)


def iter_group_maps_general_approach(X: LabelGroup, X_prime: LabelGroup) -> Generator[List[Tuple[int, int]], None, None]:
    """
    Generate all mapping from X to X_prime
    NOTE: |X| < |X_prime|

    Return mapping from (x_prime to x)
    """
    mapping_mold: List[Optional[int]] = [None for _ in range(X_prime.size)]

    for perm in permutations(range(X_prime.size), X.size):
        mapping: List[Tuple[int, int]] = []
        for i, j in enumerate(perm):
            mapping_mold[j] = X.nodes[i].id

        for i in range(X_prime.size):
            mapping.append((X_prime.nodes[i].id, mapping_mold[i]))
            mapping_mold[i] = None
        yield mapping


def iter_group_maps_using_grouping(X_prime: LabelGroup, G: List[StructureGroup]) -> Generator[List[Tuple[int, int]], None, None]:
    """
    Generate all mapping from X_prime to G (nodes in X grouped by their structures)
    NOTE: |X_prime| <= |X|

    Return mapping from (x_prime to x)
    """
    G_sizes = pvector((g.size for g in G))
    bijection: PVector = pvector([-1 for _ in range(X_prime.size)])
    terminate_index: int = X_prime.size

    call_stack: List[IterGroupMapsUsingGroupingArgs] = [
        IterGroupMapsUsingGroupingArgs(node_index=0, bijection=bijection, G_sizes=G_sizes)
    ]

    while True:
        call_args = call_stack.pop()

        if call_args.node_index == terminate_index:
            # convert bijection into a final mapping
            G_numerator = [0 for _ in range(len(G))]
            bijection = call_args.bijection
            mapping = []
            for i in range(len(bijection)):
                x_prime = X_prime.nodes[i].id
                x = G[bijection[i]].nodes[G_numerator[bijection[i]]].id

                G_numerator[bijection[i]] += 1
                mapping.append((x_prime, x))

            yield mapping
        else:
            for i, G_i in enumerate(G):
                if call_args.G_sizes[i] == 0:
                    continue

                bijection = call_args.bijection.set(call_args.node_index, i)
                G_sizes = call_args.G_sizes.set(i, call_args.G_sizes[i] - 1)

                call_stack.append(
                    IterGroupMapsUsingGroupingArgs(
                        node_index=call_args.node_index + 1, bijection=bijection, G_sizes=G_sizes))

        if len(call_stack) == 0:
            break


class DataNodeMode(IntEnum):

    NO_TOUCH = 0
    IGNORE_LABEL_DATA_NODE = 1
    IGNORE_DATA_NODE = 2


def prepare_args(gold_sm: Graph, pred_sm: Graph,
                 data_node_mode: DataNodeMode) -> List[PairLabelGroup]:
    """Prepare data for evaluation

        + data_node_mode = 0, mean we don't touch anything (note that the label of data_node must be unique)
        + data_node_mode = 1, mean we ignore label of data node (convert it to DATA_NODE, DATA_NODE2 if there are duplication columns)
        + data_node_mode = 2, mean we ignore data node
    """

    def convert_graph(graph: Graph):
        node_index: Dict[int, Node] = {}

        for v in graph.iter_nodes():
            type = Node.DATA_NODE if v.is_data_node() else Node.CLASS_NODE
            node_index[v.id] = Node(v.id, type, v.label)

        for l in graph.iter_links():
            if data_node_mode == 2:
                if node_index[l.target_id].type == Node.DATA_NODE:
                    # ignore data node
                    continue

            link = Link(l.id, l.label, l.source_id, l.target_id)
            Node.add_outgoing_link(node_index[l.source_id], link)
            Node.add_incoming_link(node_index[l.target_id], link)

        if data_node_mode == DataNodeMode.IGNORE_DATA_NODE:
            for v2 in [v for v in node_index.values() if v.type == Node.DATA_NODE]:
                del node_index[v2.id]

        if data_node_mode == DataNodeMode.IGNORE_LABEL_DATA_NODE:
            # we convert label of node to DATA_NODE
            leaf_source_nodes: Set[Node] = set()
            for v in [v for v in node_index.values() if v.type == Node.DATA_NODE]:
                assert len(v.incoming_links) == 1
                link = v.incoming_links[0]
                source = node_index[link.source_id]
                leaf_source_nodes.add(source)

            for node in leaf_source_nodes:
                link_label_count = {}
                for link in node.outgoing_links:
                    target = node_index[link.target_id]
                    if target.type == Node.DATA_NODE:
                        if link.label not in link_label_count:
                            link_label_count[link.label] = 0

                        link_label_count[link.label] += 1
                        target.label = 'DATA_NODE' + str(link_label_count[link.label])

        return node_index

    label2nodes = {}
    gold_nodes = convert_graph(gold_sm)
    pred_nodes = convert_graph(pred_sm)

    for v in gold_sm.iter_class_nodes():
        if v.label not in label2nodes:
            label2nodes[v.label] = ([], [])
        label2nodes[v.label][0].append(gold_nodes[v.id])
    for v in pred_sm.iter_class_nodes():
        if v.label not in label2nodes:
            label2nodes[v.label] = ([], [])
        label2nodes[v.label][1].append(pred_nodes[v.id])

    return [PairLabelGroup(label, LabelGroup(g[0]), LabelGroup(g[1])) for label, g in label2nodes.items()]


def f1_precision_recall(gold_sm: Graph, pred_sm: Graph, data_node_mode: DataNodeMode):
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

    bijection = Bijection.construct_from_mapping(mapping)
    list_of_dependent_groups: List[DependentGroups] = split_by_dependency(map_groups, bijection)

    best_bijections = []
    n_permutations = sum([dependent_groups.get_n_permutations() for dependent_groups in list_of_dependent_groups])

    # TODO: remove debugging code or change to logging
    if n_permutations > 50000:
        print("Number of permutation is: %d" % n_permutations)

    if n_permutations > 1000000:
        gold_sm.render2img(config.fsys.debug.tmp.as_path() + "/gold.png")
        pred_sm.render2img(config.fsys.debug.tmp.as_path() + "/pred.png")
        for dependent_groups in list_of_dependent_groups:
            print(dependent_groups.pair_groups)
        raise PermutationExploding("Cannot run evaluation because number of permutation is too high.")

    for dependent_groups in list_of_dependent_groups:
        best_bijections.append(find_best_map(dependent_groups, bijection))

    for best_bijection in best_bijections:
        bijection = bijection.extends(best_bijection)

    all_groups = DependentGroups(pair_groups)

    TP = eval_score(all_groups, bijection)
    recall = TP / max(len(all_groups.X_triples), 1)
    precision = TP / max(len(all_groups.X_prime_triples), 1)
    if TP == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # remove a useless key which causes confusion
    if None in bijection.prime2x:
        bijection.prime2x.pop(None)

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        '_bijection': bijection
    }


if __name__ == '__main__':
    dataset = "museum_crm"
    # %%

    ont = deserialize(config.fsys.debug.as_path() + '/%s/cached/ont.pkl' % dataset)
    ont_graph = deserialize(config.fsys.debug.as_path() + '/%s/cached/ont_graph.pkl' % dataset)
    karma_models = get_karma_models(dataset)

    karma_model = karma_models[59]
    # %%

    pred_sm = deserialize(config.fsys.debug.as_path() + "/tmp/pred_sm.pkl").get_semantic_model_reassign_id().graph
    gold_sm = karma_model.graph

    # %%
    res = f1_precision_recall(gold_sm, pred_sm, data_node_mode=DataNodeMode.IGNORE_LABEL_DATA_NODE)
    print(res)
    #
    # # karma_source.model.get_semantic_model_reassign_id().graph.render(120)
    # list_of_node_groups = eval(gold_sm, pred_sm, data_node_mode=1)
    # node_group = list(filter(lambda x: x[0].label == 'crm:E82_Actor_Appellation', list_of_node_groups))[0]

    # %%
    # X, X_prime = node_group
    # res = iter_group_maps_using_grouping(X_prime, LabelGroup.group_by_structures(X, X_prime))
    # res2 = list(res)
    # # %%
    # res2 = list(iter_group_maps_general_approach(X_prime, X))
    # # %%
    # ",".join([str(x) for x in node_group[0].nodes])
