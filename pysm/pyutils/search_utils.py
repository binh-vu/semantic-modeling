#!/usr/bin/python
# -*- coding: utf-8 -*-

import bisect
from typing import List, TypeVar, Optional, Callable, Generic
from typing import Tuple


T = TypeVar('T')
V = TypeVar('V')


class KeyViewArray(Generic[T, V]):

    def __init__(self, array: List[T], key: Callable[[T], V]):
        self.array = array
        self.key = key

    def __len__(self) -> int:
        return len(self.array)

    def __getitem__(self, item: int) -> V:
        return self.key(self.array[item])


def binary_search(array: List[T], value: V, key: Callable[[T], V]=None) -> T:
    """
        Do the binary search to get the index of value in the ascending sorted array.

        If the array contains one or multiple element with the value, return the first matched index.
        For example: binary_search([0, 3, 4, 4, 5], 4) will return 2.

        If the array doesn't contains the value, return the index i such that array[i-1] < value and array[i] > value
        For example:
            + binary_search([0, 3, 4, 4, 5], 2) will return 1.
            + binary_search([2, 3, 4, 4, 5], 1) will return 0.
            + binary_search([], 5) will return 0 (array[:i-1] < value is an empty array)

        In summary, the function will always return an index i satisfy: all(x < value for x in array[0:i]) and all(x >= value for x in array[i:])
        After getting the index, the following check need to be performed to know if the value is in the array is: len(array) >= 0 and i < len(array) and array[i] == value

        :param array:
        :param value:
        :param key: function to get the value
        :return: index of value in the array
    """
    if key is None:
        return bisect.bisect_left(array, value)

    return bisect.bisect_left(KeyViewArray(array, key), value)


def range_overlap_search(range_array: List[T], range_value: Tuple[float, float], key: Optional[Callable[[T], Tuple[float, float]]] = None) -> List[T]:
    """
        Search in range_array to find any range overlap with range_value. range_array is sorted ascending s.t: for any number 0 <= i < j <= len(range_array)
        range_array[i].start <= range_array[i].end <= range_array[j].start <= range_array[j].end.

        Represent range_value by tuple of (m, n), and an element in range_array (a, b). if (a, b) overlaps with (m, n) then n > a and m < b.

        The idea is:
            + search n to get an a_i, such that range_array[:a_i][0] < n
            + search m to get a b_i, such that range_array[:b_i][1] <= m (i.e range_array[b_i:][1] > m)

        The result is: range_array[b_i:a_i]

        :param range_array: if key is None, otherwise it is list of element, whose range (start, end) could be retrieve using key function
        :param range_value:
        :param key:
        :return:
    """
    def identical(x: T) -> T:
        return x

    if len(range_array) == 0:
        return []

    m, n = range_value

    if key is None:
        key = identical

    # noinspection PyTypeChecker
    a_i = binary_search(range_array, n, key=lambda x: key(x)[0])
    # noinspection PyTypeChecker
    b_i = binary_search(range_array, m, key=lambda x: key(x)[1])

    while b_i < len(range_array) and key(range_array[b_i])[1] == m:
        b_i += 1

    return range_array[b_i:a_i]
