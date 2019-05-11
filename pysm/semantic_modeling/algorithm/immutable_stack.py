#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, TypeVar, Generic

T = TypeVar('T')


class ImmutableStack(Generic[T]):

    def __init__(self) -> None:
        self.head_value: T = None
        self.size: int = 0
        self.next: ImmutableStack[T] = None

    def push(self, value: T) -> 'ImmutableStack[T]':
        if self.head_value is None:
            self.head_value = value
            self.size += 1
            return self

        stack: ImmutableStack[T] = ImmutableStack()
        stack.head_value = value
        stack.size = self.size + 1
        stack.next = self

        return stack

    def head(self) -> T:
        return self.head_value

    def pop(self):
        return self.next

    def replace(self, value: T) -> 'ImmutableStack[T]':
        assert self.head_value is not None

        stack: ImmutableStack[T] = ImmutableStack()
        stack.head_value = value
        stack.size = self.size
        stack.next = self.next
        return stack

    def __setitem__(self, key: int, value: T):
        if key < 0:
            self[self.size + key] = value
        elif key == 0:
            self.head_value = value
        else:
            self.next[key-1] = value

    def __getitem__(self, item: int) -> T:
        if item < 0:
            item = self.size + item
            return self[item]
        if item == 0:
            return self.head_value
        return self.next[item-1]
