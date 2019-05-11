#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any
import re

class RegexFilterCmd:

    def __init__(self, input_attr: str, reject_regex: str, accept_regex: str) -> None:
        self.input_attr = input_attr
        self.reject_regex = reject_regex
        self.accept_regex = accept_regex
        self.re_reject_regex = re.compile(reject_regex)
        self.re_accept_regex = re.compile(accept_regex)

    def to_dict(self):
        return {
            "_type_": "RegexFilter",
            "input_atrr": self.input_attr,
            "reject_regex": self.reject_regex,
            "accept_regex": self.accept_regex
        }

    @staticmethod
    def from_dict(obj: dict):
        return RegexFilterCmd(obj['input_attr'], obj['reject_regex'], obj['accept_regex'])