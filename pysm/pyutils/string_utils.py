#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from html.parser import HTMLParser
from typing import List, Dict, Callable, Any, Set

from pyutils.range_utils import IntervalTreeNode
from pyutils.range_utils import Range, build_interval_tree


def is_capitalize(s: str) -> bool:
    return s[0] == s[0].upper()


class Annotation(Range):
    def __init__(self, start: int, end: int):
        super(Annotation, self).__init__(start, end)

    def get_anchor(self, string: str) -> str:
        return string[self.start:self.end]

    def molding(self, string: str) -> str:
        return string

    def preview(self, text: str, window_size: int = 30, molding: bool = False, word_delimiter: str = ' ') -> str:
        """
        Preview the annotated text with the context
        """
        left_window = text[max(self.start - window_size, 0):self.start]
        anchor = text[self.start:self.end]
        right_window = text[self.end:self.end + window_size]

        if word_delimiter:
            left_window = left_window[left_window.find(word_delimiter)+1:]
            right_window = right_window[:right_window.rfind(word_delimiter)]

        if molding:
            anchor = self.molding(anchor)

        return left_window + anchor + right_window


class PolicyViolation(Exception):
    pass


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)

    def error(self, message):
        raise Exception(message)


def strip_tags(html: str) -> str:
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def find_all(substring: str, string: str) -> List[int]:
    """
        Find all occurrences of substring in string
    """
    tmp = string
    offset = 0
    occurrences = []
    while tmp.find(substring) != -1:
        occurrences.append(tmp.find(substring) + offset)
        offset += tmp.find(substring) + len(substring)
        tmp = tmp[tmp.find(substring) + len(substring):]

    return occurrences


def strseq_startswith(seq_a: List[str], seq_b: List[str], start_index: int, case_insensitive=True) -> bool:
    """Test if seqA is starts with seqB"""
    if len(seq_a) - start_index < len(seq_b):
        return False

    if case_insensitive:
        for i in range(len(seq_b)):
            if seq_a[i + start_index].lower() != seq_b[i].lower():
                return False
    else:
        for i in range(len(seq_b)):
            if seq_a[i + start_index] != seq_b[i]:
                return False

    return True


def get_match_positions(query_tokens: List[str], tokens: List[str]) -> List[int]:
    matches = []

    for i in range(len(tokens)):
        if strseq_startswith(tokens, query_tokens, i):
            matches.append(i)

    return matches


def get_hashtag(text: str) -> Set[str]:
    tags = re.findall('(#[a-zA-Z_]+)', text)
    return set(tags)


def annotate_string(string: str, annotations: List[Annotation], policy: Dict[str, Any]) -> str:
    """
        :param string: text to be annotated
        :param annotations: list of annotation need to be annotated
        :param policy: annotation policy, can be:
            NO_NESTED_ANNOTATION: default is false, the exception will be thrown if annotations are crossed or contain any other annotations
            IGNORE_NESTED_ANNOTATION: default is false. When the value is true, the contained annotation will be ignore, and keep only the surrounded annotation.
            IGNORE_CROSSED_ANNOTATION: default is true.
                + When the value is false, exception will be thrown if one annotation cross the other one.
                + When the value is true, solve conflict by removing the annotation which cause conflict in the optimum way.
            JUDGE_FUNC: when solving conflict, this function decide which node to keep and which node to discard
            MERGE_FUNC: when this function is provided, it will decide when two node are in the same range, which one to keep and which one to discard
            MERGE_ALL_FUNC: when this function is provided, it will try to merge all crossed nodes if possible, return array of new crossed nodes
        :return: annotated string
    """

    def remove_conflicted_range(node: IntervalTreeNode, judge: Callable[[Annotation, Annotation], Annotation]) -> bool:
        """
            Remove conflicted children in this node, and return a bool value indicate that
            there is any conflict has been solved. The children of the node is 
            either: crossed or separated (no containing case).

            The algorithm is very naive, it will compare 2 successive nodes and 
            choose which node to dismiss based on the judge function

            :param node:
            :param judge: True if first argument is better than the second argument
            :return: bool
        """
        if len(node.children) <= 1:
            return False

        i = 1
        children = [node.children[0]]
        while i < len(node.children):
            if children[-1].range.is_cross(node.children[i].range):
                if judge(children[-1].range, node.children[i].range):
                    # chosen node already picked, just don't do anything
                    pass
                else:
                    children[-1] = node.children[i]
            else:
                children.append(node.children[i])
            i += 1

        is_conflict = len(children) != len(node.children)
        node.children = children

        for child in node.children:
            is_conflict = remove_conflicted_range(child, judge) or is_conflict

        return is_conflict

    def annotate_substring(offset: int, string: str, annotation_node: IntervalTreeNode):
        annotation_children = annotation_node.children
        annotation = annotation_node.range

        if len(annotation_children) == 0:
            return annotation.molding(string)

        ranges = [0]
        for child in annotation_children:
            ranges.append(child.range.start - offset)
            ranges.append(child.range.end - offset)
        ranges.append(len(string))

        substrings = []
        for i, child in enumerate(annotation_children):
            substring = child.range.shift(-offset).get_anchor(string)
            substring = annotate_substring(child.range.start, substring, child)

            substrings.append(string[ranges[i * 2]:ranges[i * 2 + 1]])
            substrings.append(substring)
        substrings.append(string[ranges[-2]:ranges[-1]])

        return annotation.molding(''.join(substrings))

    if len(annotations) == 0:
        return string

    # current policy
    default_policy = {
        'NO_NESTED_ANNOTATION': False,
        'IGNORE_CROSSED_ANNOTATION': True,
        'IGNORE_NESTED_ANNOTATION': False,
        'MERGE_FUNCTION': None,
        'MERGE_ALL_FUNCTION': None,
        'JUDGE_FUNCTION': lambda a, b: True
    }
    default_policy.update(policy)
    assert len(default_policy) == 6

    # grouping annotations so that the overlapped annotations belong to same group
    # each group is sorted by the start position
    annotations.sort(key=lambda a: a.start)
    g_annotations = [[annotations[0]]]
    g_range = annotations[0]

    for i in range(1, len(annotations)):
        if g_range.is_overlapped(annotations[i]):
            g_annotations[-1].append(annotations[i])
            g_range = g_range.merge(annotations[i])
        else:
            g_annotations[-1].sort(key=lambda a: a.start)
            g_annotations.append([annotations[i]])
            g_range = annotations[i]

    if default_policy['MERGE_FUNCTION']:
        # simplify the annotated string by trying to merge 2 nodes in same range
        # to same node
        tmp_annotations = []
        for g_annotation in g_annotations:
            tmp_annotation = [g_annotation[0]]
            for i in range(1, len(g_annotation)):
                if tmp_annotation[-1].is_equal(g_annotation[i]):
                    tmp_annotation[-1] = default_policy['MERGE_FUNCTION'](tmp_annotation[-1], g_annotation[i])
                else:
                    tmp_annotation.append(g_annotation[i])
            tmp_annotations.append(tmp_annotation)
        g_annotations = tmp_annotations

    if default_policy['MERGE_ALL_FUNCTION']:
        tmp_annotations = []
        for g_annotation in g_annotations:
            tmp_annotation = default_policy['MERGE_ALL_FUNCTION'](g_annotation)
            tmp_annotations.append(tmp_annotation)
        g_annotations = tmp_annotations

    # build the annotation tree, the heart of the algorithm, which are used to render
    # the annotated string
    annotation_trees = []  # type: List[IntervalTreeNode]
    for g_annotation in g_annotations:
        if default_policy['NO_NESTED_ANNOTATION']:
            if len(g_annotation) > 1:
                raise PolicyViolation('Not allowed nested annotations')

        annotation_tree = build_interval_tree(g_annotation)
        annotation_tree.range = Annotation(annotation_tree.range.start, annotation_tree.range.end)

        if len(g_annotation) > 1:
            # noinspection PyTypeChecker
            is_conflict = remove_conflicted_range(annotation_tree, default_policy['JUDGE_FUNCTION'])
            if not default_policy['IGNORE_CROSSED_ANNOTATION'] and is_conflict:
                raise PolicyViolation('Cannot render crossed annotations')

        if default_policy['IGNORE_NESTED_ANNOTATION']:
            for child in annotation_tree.children:
                # discard all children which are nested annotation
                child.children = []

        annotation_trees.append(annotation_tree)

    # annotating the text
    ranges = [0]
    for tree in annotation_trees:
        ranges.append(tree.range.start)
        ranges.append(tree.range.end)
    ranges.append(len(string))

    annotated_strings = []
    for i, tree in enumerate(annotation_trees):
        substring = annotate_substring(
            tree.range.start,
            tree.range.get_anchor(string),
            tree
        )
        annotated_strings.append(string[ranges[i * 2]:ranges[i * 2 + 1]])
        annotated_strings.append(substring)

    annotated_strings.append(string[ranges[-2]:ranges[-1]])

    return ''.join(annotated_strings)
