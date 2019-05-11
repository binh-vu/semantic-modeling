#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
from typing import List, Tuple, TypeVar


class Range(object):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def get_id(self) -> str:
        """Get id of range: (start, end)"""
        return '%s-%s' % (self.start, self.end)

    def size(self) -> int:
        """Return size of range: end - start"""
        return self.end - self.start

    def merge(self, another: 'Range') -> 'Range':
        """Merge 2 range to become new range"""
        return Range(min(self.start, another.start), max(self.end, another.end))

    def shift(self, offset: int) -> 'Range':
        """Shift range by `offset`"""
        obj = copy.copy(self)
        obj.start += offset
        obj.end += offset
        return obj

    def is_equal(self, range: 'Range') -> bool:
        """Check if two ranges are equal"""
        return self.start == range.start and self.end == range.end

    def is_overlapped(self, another: 'Range') -> bool:
        """Check if two ranges are overlapped"""
        a, b = self, another
        if a.start > b.start:
            a, b = b, a

        return a.end > b.start

    def is_containing(self, another: 'Range') -> bool:
        """Check if this range contains `another` range"""
        return self.start <= another.start and self.end >= another.end

    def is_cross(self, another: 'Range') -> bool:
        """Check if two ranges are crossed (overlapped but not contained)"""
        return self.is_overlapped(another) and not self.is_containing(another)

    def update(self, another: 'Range') -> None:
        """Update this range with `another` range value"""
        self.start = another.start
        self.end = another.end


class IntervalTreeNode(object):
    def __init__(self, range: Range, children: List['IntervalTreeNode']):
        self.range = range
        self.children = children


def build_interval_tree(ranges: List[Range]) -> IntervalTreeNode:
    def tree_insert(node: IntervalTreeNode, range: Range):
        for child in node.children:
            if child.range.is_containing(range):
                return tree_insert(child, range)

        node.children.append(IntervalTreeNode(
            range=range,
            children=[]
        ))
        node.children.sort(key=lambda r: r.range.start)
        return

    """
        The algorithm will construct the tree such that range of subtree is contained in its parent.
        This is greedy algorithm, the algo. will sort ranges by its interval length, and insert one by one to the tree.
        
        Nodes of the returned tree are ranges, except the root of the tree is constructed by the left and right most interval.

        Examples:
            List of ranges: (1, 2), (4, 5), (1, 7), (8, 9) will result as the following tree:
                    (1, 9)
                /          \
              (1, 7)      (8, 9)
            |       |
           (1,2) (4, 5)
           List of ranges: (1, 3), (2, 3), (2, 4) will result the following tree:
                    (1, 4)
                /           \
            (1, 3)         (2, 4)
                |
            (2, 3)
    """
    left_most_value = min(ranges, key=lambda a: a.start).start
    right_most_value = max(ranges, key=lambda a: a.end).end

    root = IntervalTreeNode(Range(left_most_value, right_most_value), [])

    # sort ranges by its length, descending order, and start inserting to the tree until insert all ranges
    ranges = sorted(ranges, key=lambda a: a.end - a.start, reverse=True)
    for range in ranges:
        tree_insert(root, range)

    return root


def group_overlapped_range(ranges: List[Range]) -> List[Tuple[Range, List[Range]]]:
    if len(ranges) == 0:
        return []

    def get_range_start(x: Range) -> float:
        return x.start

    # sort by the start
    ranges = sorted(ranges, key=get_range_start)
    overlapped_ranges: List[Tuple[Range, List[Range]]] = [(Range(ranges[0].start, ranges[0].end), [ranges[0]])]

    for i in range(1, len(ranges)):
        if ranges[i].is_overlapped(overlapped_ranges[-1][0]):
            overlapped_ranges[-1][0].update(ranges[i].merge(overlapped_ranges[-1][0]))
            overlapped_ranges[-1][1].append(ranges[i])
        else:
            overlapped_ranges.append((Range(ranges[i].start, ranges[i].end), [ranges[i]]))

    return overlapped_ranges
