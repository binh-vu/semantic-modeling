#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Set, Union, Optional, Any


class SetSemanticTypeCmd:
    def __init__(self, input_attr_path: str, domain: str, type: str, node_id: str) -> None:
        self.input_attr_path = input_attr_path
        self.domain = domain
        self.type = type
        self.node_id = node_id
        assert is_correct_node_id(node_id, domain)

    def __str__(self):
        return "SetSemanticTypeCmd(attr_path=%s, domain=%s, type=%s, node_id=%s)" % (self.input_attr_path, self.domain,
                                                                                     self.type, self.node_id)

    def to_dict(self):
        return {
            "_type_": "SetSemanticType",
            "input_attr_path": self.input_attr_path,
            "node_id": self.node_id,
            "domain": self.domain,
            "type": self.type,
        }

    @staticmethod
    def from_dict(obj: dict):
        return SetSemanticTypeCmd(obj['input_attr_path'], obj['domain'], obj['type'], obj['node_id'])

    def __eq__(self, cmd: "SetSemanticTypeCmd") -> bool:
        if self is None or cmd is None or not isinstance(cmd, SetSemanticTypeCmd):
            return False

        return self.input_attr_path == cmd.input_attr_path and self.domain == cmd.domain and self.type == cmd.type and self.node_id == cmd.node_id


class SetInternalLinkCmd:
    def __init__(self, source_id: str, target_id: str, link_lbl: str, source_uri: Optional[str],
                 target_uri: Optional[str]) -> None:
        self.target_uri = target_uri
        self.source_uri = source_uri
        self.source_id = source_id
        self.target_id = target_id
        self.link_lbl = link_lbl

        assert (source_uri is None or is_correct_node_id(source_id, source_uri)) and \
               (target_uri is None or is_correct_node_id(target_id, target_uri))

    def to_dict(self):
        return {
            "_type_": "SetInternalLink",
            "target_uri": self.target_uri,
            "source_uri": self.source_uri,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_lbl": self.link_lbl
        }

    @staticmethod
    def from_dict(obj: dict):
        return SetInternalLinkCmd(obj['source_id'], obj['target_id'], obj['link_lbl'], obj['source_uri'],
                                  obj['target_uri'])

    def __eq__(self, cmd: 'SetInternalLinkCmd') -> bool:
        if self is None or cmd is None or not isinstance(cmd, SetInternalLinkCmd):
            return False

        return self.source_id == cmd.source_id and self.target_id == cmd.target_id and self.link_lbl == cmd.link_lbl and self.source_uri == cmd.source_uri and self.target_uri == cmd.target_uri

    def __str__(self):
        return "SetInternalLinkCmd(source_id=%s, link_lbl=%s, target_id=%s, source_uri=%s, target_uri=%s)" % (
            self.source_id, self.link_lbl, self.target_id, self.source_uri, self.target_uri)


def is_correct_node_id(node_id: str, node_lbl: str):
    return node_id[:-1] == node_lbl and node_id[-1:].isdigit()