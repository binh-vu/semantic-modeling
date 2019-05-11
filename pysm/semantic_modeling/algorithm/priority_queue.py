#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any, TypeVar, Generic
from heapq import *


ItemType = TypeVar('ItemType')


class PriorityTuple(Generic[ItemType]):
    def __init__(self, priority: float, item: ItemType) -> None:
        self.priority: float = priority
        self.item: ItemType = item

    def __lt__(self, other: 'PriorityTuple') -> bool:
        return self.priority < other.priority

    def __eq__(self, other: 'PriorityTuple') -> bool:
        return self.priority == other.priority


class MinPriorityQueue(Generic[ItemType]):
    """Favour low priority"""
    def __init__(self) -> None:
        self.heap: List[PriorityTuple] = []

    def push(self, priority: float, item: ItemType) -> None:
        heappush(self.heap, PriorityTuple(priority, item))

    def pop(self) -> PriorityTuple:
        return heappop(self.heap)

    def pop_item(self) -> ItemType:
        return heappop(self.heap).item

    def head(self) -> PriorityTuple:
        return self.heap[0]

    def head_item(self) -> ItemType:
        return self.heap[0].item

    def size(self) -> int:
        return len(self.heap)


class SortedListMinPriorityQueue(Generic[ItemType]):
    def __init__(self) -> None:
        self.heap: List[PriorityTuple] = []

    def push(self, priority: float, item: ItemType) -> None:
        self.heap.append(PriorityTuple(priority, item))
        self.heap.sort(reverse=True)

    def pop(self) -> PriorityTuple:
        return self.heap.pop()

    def pop_item(self) -> ItemType:
        return self.heap.pop().item

    def head(self) -> PriorityTuple:
        return self.heap[-1]

    def head_item(self) -> ItemType:
        return self.heap[-1].item

    def size(self) -> int:
        return len(self.heap)
