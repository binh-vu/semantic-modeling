#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Optional, List, Any, Union

from data_structure import Graph, GraphLinkType
from semantic_modeling.karma.karma_graph import KarmaGraph
from semantic_modeling.karma.karma_link import _dict_camel_to_snake
from semantic_modeling.karma.semantic_model import SemanticModel, Attribute
from semantic_modeling.utilities.ontology import Ontology, UselessOntology


class KarmaSourceColumn(object):

    def __init__(self, id: int, h_node_id: int, column_name: str) -> None:
        self.id = id
        self.h_node_id = h_node_id
        self.column_name: str = column_name
        assert self.id == self.h_node_id

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(obj):
        return KarmaSourceColumn(**obj)


class KarmaMappingToSourceColumn(object):

    def __init__(self, id: str, source_column_id: str) -> None:
        self.id: str = id
        self.source_column_id: str = source_column_id

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_json(obj):
        return KarmaMappingToSourceColumn(**obj)


class KarmaModel(object):

    def __init__(
        self, id: str, description: Optional[str], source_columns: List[KarmaSourceColumn],
        mapping_to_source_columns: List[KarmaMappingToSourceColumn], karma_graph: KarmaGraph
    ) -> None:
        self.id: str = id  # id of the model e.g: s12-s-19-artworks.json
        self.description: Optional[str] = description  # description
        self.original_json: Dict[str, Any] = None  # origin json serialized data of this model
        """
        source_columns is alias of Source attributes, which contains:
            id: id of the node in the graph
            hNodeId: same value as id field, however, haven't known the meaning yet
            columnName: source attribute or column in the data
        """
        self.source_columns: List[KarmaSourceColumn] = source_columns
        """haven't known the meaning yet"""
        self.mapping_to_source_columns: List[KarmaMappingToSourceColumn] = mapping_to_source_columns
        """Main graph represented the model, described carefully in KarmaGraph class"""
        self.karma_graph: KarmaGraph = karma_graph
        self.karma_graph.set_model(self)
        """A copied of karma graph, where it's highly optimized in Cython to query"""
        self.graph: Graph = self.karma_graph.to_graph()
        self.semantic_model: SemanticModel = None

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "original_json": self.original_json,
            "source_columns": [c.to_dict() for c in self.source_columns],
            "mapping_to_source_columns": [m.to_dict() for m in self.mapping_to_source_columns],
            "karma_graph": self.karma_graph.to_dict()
        }

    @staticmethod
    def from_dict(obj: dict) -> 'KarmaModel':
        model = KarmaModel(
            obj["id"], obj["description"], [KarmaSourceColumn.from_dict(o) for o in obj["source_columns"]],
            [KarmaMappingToSourceColumn.from_json(o) for o in obj["mapping_to_source_columns"]],
            KarmaGraph.from_dict(obj["karma_graph"])
        )
        return model

    @staticmethod
    def load_from_file(ont: Ontology, f_path: Union[str, Path]) -> 'KarmaModel':
        with open(f_path, 'r') as f:
            return KarmaModel.load_from_string(ont, f.read())

    @staticmethod
    def load_from_string(ont: Ontology, serialized: str) -> 'KarmaModel':
        data = json.loads(serialized)
        original_json = data
        data = _dict_camel_to_snake(data)
        """
        source_columns is alias of Source attributes, which contains:
            id: id of the node in the graph
            hNodeId: same value as id field, however, haven't known the meaning yet
            columnName: source attribute or column in the data
        """
        source_columns: List[KarmaSourceColumn] = [
            KarmaSourceColumn(**_dict_camel_to_snake(col)) for i, col in enumerate(data['source_columns'])
        ]
        """Mohsen model create an UUID for data node, and store mapping between UUID to col in mapping_to_source_columns"""
        mapping_to_source_columns: List[KarmaMappingToSourceColumn] = [
            KarmaMappingToSourceColumn(**_dict_camel_to_snake(col)) for col in data['mapping_to_source_columns']
        ]
        node_id2col = {}

        _tmp_scol = {col.id: col for col in source_columns}
        for mapping_col in mapping_to_source_columns:
            node_id2col[mapping_col.id] = _tmp_scol[mapping_col.source_column_id]
        # for col, mapping_col in zip(
        #     sorted(source_columns, key=lambda col: col.id),
        #     sorted(mapping_to_source_columns, key=lambda m: m.source_column_id)
        # ):
            # assert col.id == mapping_col.source_column_id
        """Main graph represented the model, described carefully in KarmaGraph class"""
        karma_graph, node_idmap, link_idmap = KarmaGraph.from_karma_model(
            name=data['id'].encode('utf-8'),
            graph=_dict_camel_to_snake(data['graph']),
            ontology=ont,
            id2columns=node_id2col
        )

        mapping2cols: Dict[str, str] = {c.source_column_id: c.id for c in mapping_to_source_columns}
        for col in source_columns:
            if col.id not in mapping2cols:
                col.id = -1
                col.h_node_id = -1
            else:
                col.id = node_idmap[mapping2cols[col.id]]
                col.h_node_id = node_idmap[mapping2cols[col.h_node_id]]

        model = KarmaModel(data["id"], data["description"], source_columns, mapping_to_source_columns, karma_graph)
        model.original_json = original_json

        return model

    def to_normalized_json_model(self, ont: Ontology = None) -> dict:
        """Dump the normalized/changed model back to karma model JSON format

        Few changes:
            + All id are converted from int to str so that it's compatible with source_id of link (str type due to split("---")
            + LiteralNodes is converted to ColumnNode (we are going to treat LiteralNode as a column contains only one value)
                and an new column name will be generated for LiteralNodes

        An optional ontology to restore URI from simplified version (e.g: crm:E39_Actor)
        to full version (http://www.cidoc-crm.org/cidoc-crm/E39_Actor)
        """
        nodes = []
        links = []

        if ont is None:
            ont = UselessOntology()

        # add literal nodes to source_columns
        source_columns = [{
            "id": str(col.id),
            "hNodeId": str(col.h_node_id),
            "columnName": col.column_name
        } for col in self.source_columns]
        count = len(self.source_columns)
        for node in self.karma_graph.iter_data_nodes():
            if node.is_literal_node:
                source_columns.append({
                    "id":
                    str(node.id),
                    "hNodeId":
                    str(node.id),
                    "columnName":
                    "A%d__literal_val_%s" % (count, node.label.decode('utf-8').lower().replace(" ", "-"))
                })
                count += 1
        colid2name: Dict[int, str] = {int(col['id']): col["columnName"] for col in source_columns}

        for node in self.karma_graph.iter_nodes():
            onode = {
                "id": str(node.id),
                "modelIds": None,
                "type": "InternalNode" if node.is_class_node() else "ColumnNode",
                "label": {
                    "uri": node.label.decode("utf-8")
                }
            }
            if node.is_data_node():
                onode["hNodeId"] = str(node.id)
                onode["columnName"] = colid2name[node.id]
                if node.literal_type is None:
                    onode["rdfLiteralType"] = None
                else:
                    onode["rdfLiteralType"] = {"uri": node.literal_type}

                if node.is_literal_node:
                    parent_link = node.get_first_incoming_link()
                    onode["userSemanticTypes"] = [{
                        "hNodeId": str(node.id),
                        "domain": {
                            "uri": ont.full_uri(parent_link.get_source_node().label.decode("utf-8")),
                            "rdfsLabel": None
                        },
                        "type": {
                            "uri": ont.full_uri(parent_link.label.decode("utf-8")),
                            "rdfsLabel": None
                        },
                        "origin": "User",
                        "confidenceScore": 1.0
                    }]
                    onode["learnedSemanticTypes"] = []
                else:
                    onode["userSemanticTypes"] = [{
                        "hNodeId": str(node.id),
                        "domain": {
                            "uri": ont.full_uri(st.domain),
                            "rdfsLabel": None
                        },
                        "type": {
                            "uri": ont.full_uri(st.type),
                            "rdfsLabel": None
                        },
                        "origin": st.origin,
                        "confidenceScore": st.confidence_score
                    } for st in node.user_semantic_types]
                    onode["learnedSemanticTypes"] = [{
                        "hNodeId": str(node.id),
                        "domain": {
                            "uri": ont.full_uri(st.domain),
                            "rdfsLabel": None
                        },
                        "type": {
                            "uri": ont.full_uri(st.type),
                            "rdfsLabel": None
                        },
                        "origin": "AutoModel",
                        "confidenceScore": st.confidence_score
                    } for st in node.learned_semantic_types]
            else:
                onode["label"]["uri"] = ont.full_uri(onode["label"]["uri"])

            nodes.append(onode)

        for link in self.karma_graph.iter_links():
            if link.type == GraphLinkType.OBJECT_PROPERTY:
                link_type = 'ObjectPropertyLink'
            elif link.type == GraphLinkType.DATA_PROPERTY:
                link_type = 'DataPropertyLink'
            elif link.label == 'karma:dev':
                link_type = 'ClassInstanceLink'
            elif link.get_target_node().is_data_node():
                link_type = "DataPropertyLink"
            elif link.get_target_node().is_class_node():
                link_type = "ObjectPropertyLink"

            olink = {
                "id": "%s---%s---%s" % (link.source_id, link.label.decode("utf-8"), link.target_id),
                "weight": None,
                "type": link_type,
                "label": {
                    "uri": ont.full_uri(link.label.decode("utf-8"))
                },
                "objectPropertyType": "Indirect",
                "status": "Normal",
                "keyInfo": "None",
                "modelIds": None
            }
            links.append(olink)

        model_json = {
            "id": self.id,
            "name": self.id,
            "description": self.description,
            "sourceColumns": source_columns,
            "mappingToSourceColumns": [{
                "id": col["id"],
                "sourceColumnId": col["id"]
            } for col in source_columns],
            "graph": {
                "nodes": nodes,
                "links": links
            }
        }
        return model_json

    def get_source_column_by_id(self, id: str) -> Optional[KarmaSourceColumn]:
        for col in self.source_columns:
            if col.id == id:
                return col
        return None

    def get_source_column_by_name(self, name: str) -> Optional[KarmaSourceColumn]:
        for col in self.source_columns:
            if col.column_name == name:
                return col
        return None

    def get_semantic_model(self) -> SemanticModel:
        if self.semantic_model is None:
            attrs = [Attribute(col.id, col.column_name, []) for col in self.source_columns]
            self.semantic_model = SemanticModel(self.id, attrs, self.graph)

        return self.semantic_model
