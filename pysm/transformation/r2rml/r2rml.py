#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import ujson
from pathlib import Path
import random
from typing import List, Dict, Tuple, Union

import yaml
from rdflib import Graph as RDFGraph, Namespace, BNode, RDF, Literal
from data_structure import Graph, GraphNodeType, GraphLinkType
from semantic_modeling.karma.semantic_model import Attribute, SemanticModel
from semantic_modeling.utilities.ontology import Ontology
from transformation.models.data_table import DataTable
from transformation.models.table_schema import Schema
from transformation.r2rml.commands.alter_structure import ZipAttributesCmd, UnpackOneElementListCmd, \
    AddLiteralColumnCmd, JoinListCmd
from transformation.r2rml.commands.modeling import SetSemanticTypeCmd, SetInternalLinkCmd
from transformation.r2rml.commands.pytransform import pytransform_, PyTransformNewColumnCmd, setup_modules, \
    uninstall_modules

Command = Union[PyTransformNewColumnCmd, SetInternalLinkCmd, SetSemanticTypeCmd, AddLiteralColumnCmd]


class R2RML(object):
    def __init__(self, commands: List[Command]):
        self.commands = commands

    @staticmethod
    def load_from_file(fpath: Path):
        commands = []
        if fpath.suffix == ".yaml" or fpath.suffix == ".yml":
            with open(fpath, "r") as f:
                conf = yaml.load(f)
        else:
            assert False

        for cmd in conf['commands']:
            if cmd['_type_'] == "SetInternalLink":
                commands.append(SetInternalLinkCmd.from_dict(cmd))
            elif cmd['_type_'] == 'SetSemanticType':
                commands.append(SetSemanticTypeCmd.from_dict(cmd))
            elif cmd['_type_'] == 'PyTransformNewColumn':
                commands.append(PyTransformNewColumnCmd.from_dict(cmd))
            elif cmd['_type_'] == 'ZipAttributes':
                commands.append(ZipAttributesCmd.from_dict(cmd))
            elif cmd['_type_'] == 'UnpackOneElementList':
                commands.append(UnpackOneElementListCmd.from_dict(cmd))
            elif cmd['_type_'] == 'AddLiteralColumnCmd':
                commands.append(AddLiteralColumnCmd.from_dict(cmd))
            elif cmd['_type_'] == 'JoinListCmd':
                commands.append(JoinListCmd.from_dict(cmd))
            else:
                raise NotImplementedError(f"Not support cmd: {cmd['_type_']} yet")

        return R2RML(commands)

    @staticmethod
    def load_python_scripts(python_dir: Path):
        uninstall_modules()
        setup_modules(python_dir)

    def apply_cmds(self, tbl: DataTable) -> SemanticModel:
        g = Graph(index_node_type=True, index_node_label=True, index_link_label=True, name=tbl.id.encode("utf-8"))
        attrs: List[Attribute] = []
        id_map: Dict[str, int] = {}

        for cmd in self.commands:
            if isinstance(cmd, PyTransformNewColumnCmd):
                # TODO: fix me! currently the new attr_path is generated from first input_attr_path
                # we should be explicitly about the output, since the first input attr path can be different
                # may be it should be the deepest attr path
                new_attr_path = Schema.PATH_DELIMITER.join(
                    cmd.input_attr_paths[0].split(Schema.PATH_DELIMITER)[:-1] + [cmd.new_attr_name])
                # assert not tbl.schema.has_attr_path(new_attr_path)
                # TODO: fix me!! not handle list of input attr path properly (cmd.input_attr_paths[0])
                tbl.schema.add_new_attr_path(new_attr_path, tbl.schema.get_attr_type(cmd.input_attr_paths[0]),
                                             cmd.input_attr_paths[-1])
                self.pytransform(tbl, cmd)
            elif isinstance(cmd, SetSemanticTypeCmd):
                lbl = cmd.input_attr_path.encode("utf-8")
                assert cmd.input_attr_path not in id_map
                id_map[cmd.input_attr_path] = g.add_new_node(GraphNodeType.DATA_NODE, lbl).id
                if cmd.node_id not in id_map:
                    id_map[cmd.node_id] = g.add_new_node(GraphNodeType.CLASS_NODE, cmd.domain.encode("utf-8")).id

                attrs.append(Attribute(id_map[cmd.input_attr_path], cmd.input_attr_path, []))
                g.add_new_link(GraphLinkType.UNSPECIFIED, cmd.type.encode("utf-8"), id_map[cmd.node_id],
                               id_map[cmd.input_attr_path])
            elif isinstance(cmd, SetInternalLinkCmd):
                if cmd.source_id not in id_map:
                    id_map[cmd.source_id] = g.add_new_node(GraphNodeType.CLASS_NODE, cmd.source_uri.encode('utf-8')).id
                if cmd.target_id not in id_map:
                    id_map[cmd.target_id] = g.add_new_node(GraphNodeType.CLASS_NODE, cmd.target_uri.encode('utf-8')).id

                assert g.get_node_by_id(id_map[cmd.target_id]).n_incoming_links == 0
                g.add_new_link(GraphLinkType.UNSPECIFIED, cmd.link_lbl.encode("utf-8"), id_map[cmd.source_id],
                               id_map[cmd.target_id])
            elif isinstance(cmd, ZipAttributesCmd):
                for row in tbl.rows:
                    cmd.zip_attributes(row)
                # TODO: fix me!! re-build schema, which is very expensive
                tbl.rebuild_schema()
            elif isinstance(cmd, UnpackOneElementListCmd):
                assert tbl.schema.get_attr_type(cmd.input_attr) == Schema.LIST_VALUE
                for row in tbl.rows:
                    cmd.unpack(row)
                tbl.schema.update_attr_path(cmd.input_attr, Schema.SINGLE_VALUE)
            elif isinstance(cmd, AddLiteralColumnCmd):
                tbl.schema.add_new_attr_path(cmd.input_attr_path, tbl.schema.SINGLE_VALUE)
                for row in tbl.rows:
                    cmd.add_literal(row)
            elif isinstance(cmd, JoinListCmd):
                for row in tbl.rows:
                    cmd.execute(row)
                tbl.schema.update_attr_path(cmd.input_attr_path, Schema.SINGLE_VALUE)
            else:
                raise NotImplementedError(cmd.__class__.__name__)

        return SemanticModel(tbl.id, attrs, g)

    def apply_build(self, tbl: DataTable) -> Tuple[DataTable, SemanticModel]:
        sm = self.apply_cmds(tbl)
        schema = tbl.schema.clone()
        for attr_path in set(tbl.schema.get_attr_paths()).difference((attr.label for attr in sm.attrs)):
            schema.delete_attr_path(attr_path)
        assert len(schema.get_attr_paths()) == len(sm.attrs)

        new_tbl = DataTable(tbl.id, schema, [schema.normalize(r) for r in tbl.rows])
        return new_tbl, sm

    @staticmethod
    def pytransform(tbl: DataTable, cmd: PyTransformNewColumnCmd):
        """May takes multiple input columns, and produce one output columns, currently assume that we all input columns
        share same prefix path, and only the last attribute name is changed, e.g: artist.name, artist.address
        """
        prefix_attr_path = None
        input_attrs = []
        for input_attr_path in cmd.input_attr_paths:
            attr_path = input_attr_path.split(Schema.PATH_DELIMITER)
            if prefix_attr_path is None:
                prefix_attr_path = Schema.PATH_DELIMITER.join(attr_path[:-1])
            else:
                assert prefix_attr_path == Schema.PATH_DELIMITER.join(attr_path[:-1])
            input_attrs.append(attr_path[-1])

        if prefix_attr_path == "":
            prefix_sep_path = prefix_attr_path
            prefix_attr_path = []
        else:
            prefix_sep_path = prefix_attr_path + Schema.PATH_DELIMITER
            prefix_attr_path = prefix_attr_path.split(Schema.PATH_DELIMITER)

        for row in tbl.rows:
            pytransform_(row, prefix_sep_path, row, tbl.schema, prefix_attr_path, input_attrs, cmd.new_attr_name,
                         cmd.code)

    def to_dict(self):
        return {
            "commands": [cmd.to_dict() for cmd in self.commands]
        }

    def to_yaml(self, fpath: Path):
        with open(fpath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=4)

    def reformat_yaml(self, fpath: Path) -> 'R2RML':
        commands = []
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, AddLiteralColumnCmd))
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, (PyTransformNewColumnCmd, ZipAttributesCmd, UnpackOneElementListCmd)))
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, SetSemanticTypeCmd))
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, SetInternalLinkCmd))
        assert len(commands) == len(self.commands)
        
        with open(fpath, "w") as f:
            yaml.dump({"commands": commands}, f, default_flow_style=False, indent=4)

        return self

    def to_kr2rml(self, ont: Ontology, tbl: DataTable, fpath: Union[str, Path]):
        g = RDFGraph()
        km_dev = Namespace("http://isi.edu/integration/karma/dev#")
        g.namespace_manager.bind("km-dev", km_dev)
        kr2rml = BNode()

        g.add((kr2rml, RDF.type, km_dev.R2RMLMapping))
        g.add((kr2rml, km_dev.sourceName, Literal(tbl.id)))
        # timestamp and version, doesn't need to be precise
        g.add((kr2rml, km_dev.modelPublicationTime, Literal(1414133381264)))
        g.add((kr2rml, km_dev.modelVersion, Literal("1.7")))

        input_columns = []
        output_columns = []
        # mapping from Schema attribute path OR Command to KarmaColumns
        attr2hnodes: Dict[Union[str, PyTransformNewColumnCmd], List[Dict[str, str]]] = {}

        for attr_path in tbl.schema.get_attr_paths():
            input_columns.append([{"columnName": x} for x in attr_path.split(Schema.PATH_DELIMITER)])
            if tbl.schema.get_attr_type(attr_path) == Schema.LIST_VALUE:
                # default karma behaviour, you cannot set semantic type for higher level, but only "values"
                input_columns[-1].append({"columnName": "values"})
            output_columns.append(input_columns[-1])
            attr2hnodes[attr_path] = input_columns[-1]

        for cmd in self.commands:
            if isinstance(cmd, PyTransformNewColumnCmd):
                new_attr_path = cmd.input_attr_paths[0].split(Schema.PATH_DELIMITER)[:-1]
                new_attr_path.append(cmd.new_attr_name)
                new_attr_path = Schema.PATH_DELIMITER.join(new_attr_path)

                # when you create a new column from a list, karma convert to a list of objects
                # e.g: birth_death_date.values, create col death date from that,
                # Karma create => birth_death_date.death_date
                # that's why we have this code below
                new_hnode = attr2hnodes[cmd.input_attr_paths[0]][:-1]
                new_hnode.append({"columnName": cmd.new_attr_name})

                output_columns.append(new_hnode)
                attr2hnodes[cmd] = output_columns[-1]
                attr2hnodes[new_attr_path] = output_columns[-1]

        worksheet_history = []
        # re-arrange commands to fit the issue of node id = Concept2 (Karma will convert Concept2 to Concept1)
        commands = [cmd for cmd in self.commands if isinstance(cmd, PyTransformNewColumnCmd)]
        for cmd in sorted([c for c in self.commands if isinstance(c, SetSemanticTypeCmd)], key=lambda c: c.node_id):
            commands.append(cmd)

        for cmd in sorted([c for c in self.commands if isinstance(c, SetInternalLinkCmd)], key=lambda c: c.target_uri or c.source_uri or ""):
            commands.append(cmd)
        
        # sometime the model use incorrect node id like: node id = Concept7 (no Concept1..6), will result as an error in Karma
        # need to re-arrange the node_id
        node_id_old2new: Dict[str, str] = {}
        node_id_domain_count: Dict[str, int] = {}

        for cmd in commands:
            if isinstance(cmd, PyTransformNewColumnCmd):
                pass
            elif isinstance(cmd, SetSemanticTypeCmd):
                if cmd.node_id not in node_id_old2new:
                    node_id_domain_count[cmd.domain] = node_id_domain_count.get(cmd.domain, 0) + 1
                    node_id_old2new[cmd.node_id] = f"{cmd.domain}{node_id_domain_count[cmd.domain]}"
            elif isinstance(cmd, SetInternalLinkCmd):
                if cmd.source_id not in node_id_old2new:
                    assert cmd.source_uri is not None
                    node_id_domain_count[cmd.source_uri] = node_id_domain_count.get(cmd.source_uri, 0) + 1
                    node_id_old2new[cmd.source_id] = f"{cmd.source_uri}{node_id_domain_count[cmd.source_uri]}"
                if cmd.target_id not in node_id_old2new:
                    assert cmd.target_uri is not None
                    node_id_domain_count[cmd.target_uri] = node_id_domain_count.get(cmd.target_uri, 0) + 1
                    node_id_old2new[cmd.target_id] = f"{cmd.target_uri}{node_id_domain_count[cmd.target_uri]}"

        for cmd in commands:
            if isinstance(cmd, PyTransformNewColumnCmd):
                pytransform_code = cmd.code
                # recover pytransform_code from our code
                pytransform_code = pytransform_code.replace("__return__ = ", "return ")
                for match in reversed(list(re.finditer("getValue\(([^)]+)\)", pytransform_code))):
                    start, end = match.span(1)
                    field = pytransform_code[start:end].replace("'", "").replace('"""', "").replace('"', '')
                    # convert full name to last column name since Karma use last column name instead
                    for input_attr_path in cmd.input_attr_paths:
                        if input_attr_path == field:
                            # TODO: will Karma always use last column name?
                            field = attr2hnodes[input_attr_path][-1]['columnName']
                            break
                    else:
                        assert False, f"Cannot find any field {field} in the input columns"
                    pytransform_code = pytransform_code[:start] + f'"{field}"' + pytransform_code[end:]

                worksheet_history.append({
                    "tags": ["Transformation"],
                    "commandName": "SubmitPythonTransformationCommand",
                    "inputParameters": [
                        {"name": "hNodeId", "value": attr2hnodes[cmd.input_attr_paths[0]], "type": "hNodeId"},
                        {"name": "worksheetId", "value": "W", "type": "worksheetId"},
                        {"name": "selectionName", "value": "DEFAULT_TEST", "type": "other"},
                        {"name": "newColumnName", "value": cmd.new_attr_name, "type": "other"},
                        {"name": "transformationCode", "value": pytransform_code, "type": "other"},
                        {"name": "errorDefaultValue", "value": cmd.default_error_value, "type": "other"},
                        {
                            "name": "inputColumns",
                            "type": "hNodeIdList",
                            "value": ujson.dumps([{"value": attr2hnodes[iap]} for iap in cmd.input_attr_paths])
                        },
                        {
                            "name": "outputColumns",
                            "type": "hNodeIdList",
                            "value": ujson.dumps([{"value": attr2hnodes[cmd] if attr2hnodes[cmd][-1]['columnName'] != "values" else attr2hnodes[cmd][:-1]}])
                        }
                    ]
                })
            elif isinstance(cmd, SetSemanticTypeCmd):
                if cmd.type != "karma:classLink":
                    worksheet_history.append({
                        "commandName": "SetSemanticTypeCommand",
                        "tags": ["Modeling"],
                        "inputParameters": [
                            {"name": "hNodeId", "value": attr2hnodes[cmd.input_attr_path], "type": "hNodeId"},
                            {"name": "worksheetId", "value": "W", "type": "worksheetId"},
                            {"name": "selectionName", "value": "DEFAULT_TEST", "type": "other"},
                            {
                                "name": "SemanticTypesArray",
                                "type": "other",
                                "value": [{
                                    "FullType": ont.full_uri(cmd.type),
                                    "isPrimary": True,
                                    "DomainLabel": ont.simplify_uri(node_id_old2new[cmd.node_id]),
                                    "DomainId": ont.full_uri(node_id_old2new[cmd.node_id]),
                                    "DomainUri": ont.full_uri(cmd.domain)
                                }]
                            },
                            {"name": "trainAndShowUpdates", "value": False, "type": "other"},
                            {"name": "rdfLiteralType", "value": "", "type": "other"},  # TODO: update correct RDF-Literal-Type
                            {
                                "name": "inputColumns",
                                "type": "hNodeIdList",
                                "value": ujson.dumps([{"value": attr2hnodes[cmd.input_attr_path]}])
                            },
                            {
                                "name": "outputColumns",
                                "type": "hNodeIdList",
                                "value": ujson.dumps([{"value": attr2hnodes[cmd.input_attr_path]}])
                            }
                        ]
                    })
                else:
                    worksheet_history.append({
                        "commandName": "SetMetaPropertyCommand",
                        "tags": ["Modeling"],
                        "inputParameters": [
                            {"name": "hNodeId", "value": attr2hnodes[cmd.input_attr_path], "type": "hNodeId"},
                            {"name": "worksheetId", "value": "W", "type": "worksheetId"},
                            {"name": "selectionName", "value": "DEFAULT_TEST", "type": "other"},
                            {"name": "metaPropertyName", "value": "isUriOfClass", "type": "other"},
                            {"name": "metaPropertyUri", "value": ont.full_uri(cmd.domain), "type": "other"},
                            {"name": "metaPropertyId", "value": ont.full_uri(node_id_old2new[cmd.node_id]), "type": "other"},
                            {
                                "name": "SemanticTypesArray",
                                "type": "other",
                                "value": [{
                                    "FullType": ont.full_uri(cmd.type),
                                    "isPrimary": True,
                                    "DomainLabel": ont.simplify_uri(node_id_old2new[cmd.node_id]),
                                    "DomainId": ont.full_uri(node_id_old2new[cmd.node_id]),
                                    "DomainUri": ont.full_uri(cmd.domain)
                                }]
                            },
                            {"name": "trainAndShowUpdates", "value": False, "type": "other"},
                            {"name": "rdfLiteralType", "value": "", "type": "other"}, # TODO: update correct RDF-Literal-Type
                            {
                                "name": "inputColumns",
                                "type": "hNodeIdList",
                                "value": ujson.dumps([{"value": attr2hnodes[cmd.input_attr_path]}])
                            },
                            {
                                "name": "outputColumns",
                                "type": "hNodeIdList",
                                "value": ujson.dumps([{"value": attr2hnodes[cmd.input_attr_path]}])
                            }
                        ]
                    })
            elif isinstance(cmd, SetInternalLinkCmd):
                # TODO: comment out because old KARMA doesn't recognize this!
                # if cmd.target_uri is not None or cmd.source_uri is not None:
                #     worksheet_history.append({
                #         "commandName": "AddLinkCommand",
                #         "tags": ["Modeling"],
                #         "inputParameters": [
                #             {"name": "worksheetId", "value": "W", "type": "worksheetId"},
                #             {
                #                 "name": "edge",
                #                 "type": "other",
                #                 "value": {
                #                     "edgeId": ont.full_uri(cmd.link_lbl),
                #                     "edgeTargetId": ont.full_uri(node_id_old2new[cmd.target_id]),
                #                     "edgeTargetUri": ont.full_uri(cmd.target_uri or cmd.target_id[:-1]),
                #                     "edgeSourceId": ont.full_uri(node_id_old2new[cmd.source_id]),
                #                     "edgeSourceUri": ont.full_uri(cmd.source_uri or cmd.source_id[:-1])
                #                 }
                #             },
                #             {"name": "inputColumns", "type": "hNodeIdList", "value": []},
                #             {"name": "outputColumns", "type": "hNodeIdList", "value": []}
                #         ]
                #     })
                # else:
                worksheet_history.append({
                    "commandName": "ChangeInternalNodeLinksCommand",
                    "tags": ["Modeling"],
                    "inputParameters": [
                        {"name": "worksheetId", "value": "W", "type": "worksheetId"},
                        {
                            "name": "initialEdges",
                            "type": "other",
                            "value": [{
                                "edgeId": ont.full_uri(cmd.link_lbl),
                                "edgeTargetId": ont.full_uri(node_id_old2new[cmd.target_id]),
                                "edgeSourceId": ont.full_uri(node_id_old2new[cmd.source_id])
                            }]
                        },
                        {
                            "name": "newEdges",
                            "type": "other",
                            "value": [{
                                "edgeId": ont.full_uri(cmd.link_lbl),
                                "edgeTargetId": ont.full_uri(node_id_old2new[cmd.target_id]),
                                "edgeSourceId": ont.full_uri(node_id_old2new[cmd.source_id]),
                                "edgeTargetUri": ont.full_uri(cmd.target_uri or node_id_old2new[cmd.target_id][:-1]),
                                "edgeSourceUri": ont.full_uri(cmd.source_uri or node_id_old2new[cmd.source_id][:-1])
                            }]
                        },
                        {"name": "inputColumns", "type": "hNodeIdList", "value": []},
                        {"name": "outputColumns", "type": "hNodeIdList", "value": []}
                    ]
                })

        g.add((kr2rml, km_dev.hasInputColumns, Literal(ujson.dumps(input_columns))))
        g.add((kr2rml, km_dev.hasOutputColumns, Literal(ujson.dumps(output_columns))))
        g.add((kr2rml, km_dev.hasModelLabel, Literal(tbl.id)))
        g.add((kr2rml, km_dev.hasBaseURI, Literal("http://localhost:8080/source/")))
        g.add((kr2rml, km_dev.hasWorksheetHistory, Literal(ujson.dumps(worksheet_history, indent=4))))

        g.serialize(str(fpath), format='n3')


if __name__ == '__main__':
    from semantic_modeling.data_io import get_ontology

    ont = get_ontology("museum_edm")
    r2rml = R2RML.load_from_file(Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/models-y2rml/s01-cb-model.yml"))
    tbl = DataTable.load_from_file(Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/sources/s01-cb.csv"))
    r2rml.to_kr2rml(ont, tbl, Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/karma-version/models-r2rml/s01-cb-model.ttl"))

    # r2rml = R2RML.load_from_file(Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/models-y2rml/s04-ima-artworks-model.yml"))
    # tbl = DataTable.load_from_file(Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/sources/s04-ima-artworks.xml"))
    # r2rml.to_kr2rml(ont, tbl, Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/karma-version/models-r2rml/s04-ima-artworks-model.ttl"))

    # r2rml = R2RML.load_from_file(Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/models-y2rml/s09-s-18-artists-model.yml"))
    # tbl = DataTable.load_from_file(Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/sources/s09-s-18-artists.json"))
    # r2rml.to_kr2rml(ont, tbl, Path("/home/rook/workspace/DataIntegration/SourceModeling/data/museum-edm/models-r2rml/s09-s-18-artists-model.ttl"))

    # res, sm = r2rml.apply_build(tbl)
    # print(res.head(5).to_string())
    # print([a.id for a in sm.attrs])
    # sm.graph.render()

