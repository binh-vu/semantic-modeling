#!/usr/bin/python
# -*- coding: utf-8 -*-
import ujson, re
from collections import defaultdict, OrderedDict
from pathlib import Path

from typing import Dict, Tuple, List, Set, Union, Optional

import rdflib
import yaml
from rdflib import URIRef
from rdflib.plugins.memory import IOMemory

from semantic_modeling.config import get_logger
from semantic_modeling.data_io import get_ontology
from semantic_modeling.utilities.ontology import Ontology
from transformation.models.data_table import DataTable
from transformation.models.table_schema import Schema
from transformation.r2rml.commands.alter_structure import UnrollCmd, AddLiteralColumnCmd
from transformation.r2rml.commands.modeling import SetSemanticTypeCmd, SetInternalLinkCmd
from transformation.r2rml.commands.pytransform import PyTransformNewColumnCmd
from transformation.r2rml.r2rml import R2RML


class KR2RML(R2RML):
    """Load KR2RML and produce default command history"""
    logger = get_logger("app.transformation.kr2rml")

    def __init__(self, ont: Ontology, tbl: DataTable, kr2rml_file: Path) -> None:
        g = rdflib.Graph(store=IOMemory())
        g.parse(location=str(kr2rml_file), format="n3")

        worksheet_history = list(
            g.triples((None, URIRef("http://isi.edu/integration/karma/dev#hasWorksheetHistory"), None)))
        assert len(worksheet_history) == 1
        worksheet_history = ujson.loads(worksheet_history[0][-1])

        input_columns = list(g.triples((None, URIRef("http://isi.edu/integration/karma/dev#hasInputColumns"), None)))
        assert len(input_columns) == 1
        input_columns = ujson.loads(input_columns[0][-1])

        # construct mapping between kr2rml attribute paths to tbl_attr_paths
        tbl_attr_paths = tbl.schema.get_attr_paths()
        n_attr_paths = len(tbl_attr_paths)
        tbl_attr_paths = {apath.replace("@", ""): apath for apath in tbl_attr_paths}
        assert len(tbl_attr_paths) == n_attr_paths

        start_idx = 0
        for i, cname in enumerate(input_columns[0]):
            cpath = Schema.PATH_DELIMITER.join(cname['columnName'] for cname in input_columns[0][i:])
            # cname = Schema.PATH_DELIMITERinput_columns[i:]) cname['columnName'] + Schema.PATH_DELIMITER
            found_attr = False
            for attr_path in tbl_attr_paths:
                if (attr_path + Schema.PATH_DELIMITER).startswith(cpath):
                    found_attr = True
                    break
            if found_attr:
                start_idx = i
                break

        literal_nodes = {}
        col2col = {}
        for col in input_columns:
            attr_path = Schema.PATH_DELIMITER.join(cname['columnName'] for cname in col[start_idx:])
            if attr_path not in tbl_attr_paths:
                attr_path = Schema.PATH_DELIMITER.join(cname['columnName'] for cname in col[start_idx:-1])
                if col[-1]['columnName'] == 'Values':
                    assert attr_path in tbl_attr_paths
                elif col[-1]['columnName'] == 'content':
                    attr_path += Schema.PATH_DELIMITER + "#text"
                    assert attr_path in tbl_attr_paths
                else:
                    raise ValueError(f"Invalid column type: {col[-1]['columnName']}")

            col2col[Schema.PATH_DELIMITER.join(cname['columnName'] for cname in col)] = tbl_attr_paths[attr_path]
        assert len(set(col2col.values())) == len(input_columns), "No duplication"

        # extracting commands
        commands = []
        for command in worksheet_history:
            if command['commandName'] == "SubmitPythonTransformationCommand":
                cmd_start_col = command['inputParameters'][0]
                cmd_input_parent_col = Schema.PATH_DELIMITER.join([col['columnName'] for col in cmd_start_col['value'][:-1]])
                cmd_input_col = command['inputParameters'][-2]
                cmd_output_col = command['inputParameters'][-1]

                if command['inputParameters'][-3]['name'] == 'isJSONOutput':
                    cmd_code = command['inputParameters'][-5]
                    default_error_value = command['inputParameters'][-4]
                    assert command['inputParameters'][-3]['value'] == "false"
                else:
                    default_error_value = command['inputParameters'][-3]
                    cmd_code = command['inputParameters'][-4]

                assert cmd_input_col['name'] == "inputColumns" and cmd_output_col["name"] == "outputColumns" and cmd_code['name'] == 'transformationCode' and default_error_value['name'] == 'errorDefaultValue'
                cmd_input_cols = [
                    [cname['columnName'] for cname in o['value']] for o in ujson.loads(cmd_input_col['value'])
                ]
                karma_input_attr_paths = [col2col[Schema.PATH_DELIMITER.join(cmd_input_col)] for cmd_input_col in
                                    cmd_input_cols]

                # update col2col because of new columns
                new_attr_name = ujson.loads(cmd_output_col['value'])[0]['value'][-1]['columnName']
                new_attr_path = new_attr_name if cmd_input_parent_col == "" else (cmd_input_parent_col + Schema.PATH_DELIMITER + new_attr_name)
                cmd_output_col = Schema.PATH_DELIMITER.join(
                    cname['columnName'] for cname in ujson.loads(cmd_output_col['value'])[0]['value'])
                col2col[cmd_output_col] = new_attr_path

                cmd_code = cmd_code['value'].replace("return ", "__return__ = ")
                input_attr_paths = []
                for match in reversed(list(re.finditer("getValue\(([^)]+)\)", cmd_code))):
                    start, end = match.span(1)
                    field = cmd_code[start:end].replace("'", "").replace('"""', "").replace('"', '')
                    # it seems that Karma use last column name, we need to recover full name
                    # using the provided input first
                    for cmd_input_col, input_attr_path in zip(cmd_input_cols, karma_input_attr_paths):
                        if field == cmd_input_col[-1]:
                            field = input_attr_path
                            break
                    else:
                        # otherwise construct from the start columns
                        full_field = field if cmd_input_parent_col == "" else (cmd_input_parent_col + Schema.PATH_DELIMITER + field)
                        field = col2col[full_field]
                    cmd_code = cmd_code[:start] + f'"{field}"' + cmd_code[end:]

                    input_attr_paths.append(field)

                default_error_value = default_error_value['value']
                commands.append(PyTransformNewColumnCmd(input_attr_paths, new_attr_name, cmd_code, default_error_value))
            elif command["commandName"] == "SetSemanticTypeCommand" or command["commandName"] == "SetMetaPropertyCommand":
                cmd_input_col = command['inputParameters'][-2]
                if command["inputParameters"][-5]['name'] == 'SemanticTypesArray':
                    cmd_stype = command['inputParameters'][-5]
                else:
                    cmd_stype = command['inputParameters'][-6]

                if cmd_stype['name'] == 'SemanticTypesArray':
                    assert cmd_input_col['name'] == "inputColumns" and len(
                        cmd_stype['value']) == 1 and cmd_stype['value'][0]['isPrimary']
                    cmd_input_col = col2col[Schema.PATH_DELIMITER.join(
                        cname['columnName'] for cname in ujson.loads(cmd_input_col['value'])[0]['value'])]
                    cmd_stype = cmd_stype['value'][0]

                    commands.append(
                        SetSemanticTypeCmd(
                            cmd_input_col,
                            domain=ont.simplify_uri(cmd_stype['DomainUri']),
                            type=ont.simplify_uri(cmd_stype['FullType']),
                            node_id=ont.simplify_uri(cmd_stype['DomainId'].replace(" (add)", ""))))
                else:
                    cmd_stype_domain = command['inputParameters'][-7]
                    cmd_stype_id = command['inputParameters'][-6]
                    assert cmd_input_col['name'] == "inputColumns" and cmd_stype_domain['name'] == 'metaPropertyUri' \
                           and cmd_stype_id['name'] == 'metaPropertyId'
                    cmd_input_col = col2col[Schema.PATH_DELIMITER.join(
                        cname['columnName'] for cname in ujson.loads(cmd_input_col['value'])[0]['value'])]

                    commands.append(
                        SetSemanticTypeCmd(
                            cmd_input_col,
                            domain=ont.simplify_uri(cmd_stype_domain['value']),
                            type="karma:classLink",
                            node_id=ont.simplify_uri(cmd_stype_id['value'])))
            elif command['commandName'] == 'UnassignSemanticTypeCommand':
                cmd_input_col = command['inputParameters'][-2]
                assert cmd_input_col['name'] == "inputColumns"
                cmd_input_col = col2col[Schema.PATH_DELIMITER.join(
                    cname['columnName'] for cname in ujson.loads(cmd_input_col['value'])[0]['value'])]

                delete_cmds = []
                for i, cmd in enumerate(commands):
                    if isinstance(cmd, SetSemanticTypeCmd) and cmd.input_attr_path == cmd_input_col:
                        delete_cmds.append(i)

                for i in reversed(delete_cmds):
                    commands.pop(i)
            elif command["commandName"] == "ChangeInternalNodeLinksCommand":
                cmd_edges = command['inputParameters'][-3]
                assert cmd_edges['name'] == 'newEdges'
                # cmd_initial_edges = command['inputParameters'][-4]
                # if cmd_initial_edges['name'] == 'initialEdges' and len(cmd_initial_edges['value']) > 0:
                #     delete_cmds = []
                #     for cmd_edge in cmd_initial_edges['value']:
                #         edge_lbl = ont.simplify_uri(cmd_edge['edgeId'])
                #         source_id = ont.simplify_uri(cmd_edge['edgeSourceId'])
                #
                #         if cmd_edge['edgeTargetId'] in literal_nodes:
                #             for i, cmd in enumerate(commands):
                #                 if isinstance(cmd, SetSemanticTypeCmd) and cmd.type == edge_lbl and cmd.node_id == source_id:
                #                         delete_cmds.append(i)
                #         else:
                #             target_id = ont.simplify_uri(cmd_edge['edgeTargetId'])
                #             for i, cmd in enumerate(commands):
                #                 if isinstance(cmd, SetInternalLinkCmd) and cmd.link_lbl == edge_lbl and cmd.target_id == target_id and cmd.source_id == source_id:
                #                     delete_cmds.append(i)
                #
                #     for idx in sorted(delete_cmds, reverse=True):
                #         commands.pop(idx)

                for cmd_edge in cmd_edges['value']:
                    source_uri = cmd_edge.get('edgeSourceUri', None)
                    target_uri = cmd_edge.get('edgeTargetUri', None)

                    if source_uri is not None and source_uri != cmd_edge['edgeSourceId']:
                        source_uri = ont.simplify_uri(source_uri)
                    else:
                        source_uri = None

                    if target_uri is not None and target_uri != cmd_edge['edgeTargetId']:
                        target_uri = ont.simplify_uri(target_uri)
                    else:
                        target_uri = None

                    if cmd_edge['edgeTargetId'] in literal_nodes:
                        # convert this command to SetSemanticType
                        commands.append(SetSemanticTypeCmd(
                            literal_nodes[cmd_edge['edgeTargetId']],
                            domain=ont.simplify_uri(source_uri),
                            type=ont.simplify_uri(cmd_edge['edgeId']),
                            node_id=ont.simplify_uri(cmd_edge['edgeSourceId'])
                        ))
                    else:
                        commands.append(
                            SetInternalLinkCmd(ont.simplify_uri(cmd_edge['edgeSourceId']), ont.simplify_uri(cmd_edge['edgeTargetId']),
                                               ont.simplify_uri(cmd_edge['edgeId']),
                                               source_uri, target_uri))
            elif command['commandName'] == "AddLinkCommand":
                cmd_edges = command['inputParameters'][-3]
                assert cmd_edges['name'] == 'edge'
                cmd_edge = cmd_edges['value']
                source_uri = cmd_edge.get('edgeSourceUri', None)
                target_uri = cmd_edge.get('edgeTargetUri', None)
                if source_uri is not None:
                    source_uri = ont.simplify_uri(source_uri)
                else:
                    source_uri = None

                if cmd_edge['edgeTargetId'] in literal_nodes:
                    # convert this command to SetSemanticType
                    commands.append(SetSemanticTypeCmd(
                        literal_nodes[cmd_edge['edgeTargetId']],
                        domain=ont.simplify_uri(source_uri),
                        type=ont.simplify_uri(cmd_edge['edgeId']),
                        node_id=ont.simplify_uri(cmd_edge['edgeSourceId'])
                    ))
                else:
                    if target_uri is not None:
                        target_uri = ont.simplify_uri(target_uri)
                    else:
                        target_uri = None

                    commands.append(
                        SetInternalLinkCmd(ont.simplify_uri(cmd_edge['edgeSourceId']), ont.simplify_uri(cmd_edge['edgeTargetId']),
                                           ont.simplify_uri(cmd_edge['edgeId']),
                                           source_uri, target_uri))
            elif command['commandName'] == 'DeleteLinkCommand':
                cmd_edge = command['inputParameters'][-3]
                assert cmd_edge['name'] == 'edge'
                cmd_edge = cmd_edge['value']
                for i, cmd in enumerate(commands):
                    if isinstance(cmd, SetInternalLinkCmd):
                        if cmd.source_id == cmd_edge['edgeSourceId'] and cmd.target_id == cmd_edge['edgeTargetId'] and cmd.link_lbl == ont.simplify_uri(cmd_edge['edgeId']):
                            commands.pop(i)
                            break
            elif command["commandName"] == "AddLiteralNodeCommand":
                cmd_literal_value = command["inputParameters"][0]
                assert cmd_literal_value['name'] == 'literalValue'
                cmd_literal_value = cmd_literal_value['value']

                # they may re-use literal_values, let's user fix it manually
                if cmd_literal_value.startswith("http"):
                    new_attr_path = f"literal:{ont.simplify_uri(cmd_literal_value)}"
                else:
                    new_attr_path = f"literal:{cmd_literal_value}"

                if cmd_literal_value + "1" not in literal_nodes:
                    new_attr_path += ":1"
                    literal_nodes[cmd_literal_value + "1"] = new_attr_path
                elif cmd_literal_value + "2" not in literal_nodes:
                    new_attr_path += ":2"
                    literal_nodes[cmd_literal_value + "2"] = new_attr_path
                elif cmd_literal_value + "3" not in literal_nodes:
                    new_attr_path += ":3"
                    literal_nodes[cmd_literal_value + "3"] = new_attr_path
                else:
                    assert False

                col2col[new_attr_path] = new_attr_path
                commands.append(AddLiteralColumnCmd(new_attr_path, cmd_literal_value))
            elif command["commandName"] == "OperateSelectionCommand":
                # no way to see it in the KARMA UI
                continue
            elif command["commandName"] == "OrganizeColumnsCommand":
                continue
            elif command["commandName"] == "SetWorksheetPropertiesCommand":
                # this command doesn't affect the model
                continue
            # elif command["commandName"] == "UnfoldCommand":
            #     cmd_input_col = command["inputParameters"][-2]
            #     cmd_output_col = command["inputParameters"][-1]
            #     assert cmd_input_col['name'] == "inputColumns" and cmd_output_col['name'] == 'outputColumns'
            #     cmd_input_cols = [
            #         [cname['columnName'] for cname in o['value']] for o in ujson.loads(cmd_input_col['value'])
            #     ]
            #     input_attr_paths = [col2col[Schema.PATH_DELIMITER.join(cmd_input_col)] for cmd_input_col in cmd_input_cols]
            #     cmd_output_cols = [
            #         [cname['columnName'] for cname in o['value']] for o in ujson.loads(cmd_output_col['value'])
            #     ]
            #
            #     output_attr_paths = []
            #     # update columns mapping
            #     for cmd_output_col in cmd_output_cols:
            #         attr_path = Schema.PATH_DELIMITER.join(cmd_output_col[start_idx:])
            #         col2col[Schema.PATH_DELIMITER.join(cmd_output_col)] = attr_path
            #         output_attr_paths.append(attr_path)
            #
            #     commands.append(UnrollCmd(input_attr_paths, output_attr_paths))
            # elif command["commandName"] == "GlueCommand":
            #     cmd_input_col = command["inputParameters"][-2]
            #     cmd_output_col = command["inputParameters"][-1]
            else:
                assert False, "Source: %s. Doesn't handle command %s" % (tbl.id, command["commandName"])

        # fixing conflict modeling command
        conflicts = defaultdict(lambda: [])
        for i, cmd in enumerate(commands):
            if isinstance(cmd, SetSemanticTypeCmd):
                conflicts[cmd.input_attr_path].append((i, cmd))
            if isinstance(cmd, SetInternalLinkCmd):
                conflicts[(cmd.source_id, cmd.target_id)].append((i, cmd))

        delete_commands = []
        for cmds in conflicts.values():
            if len(cmds) > 1:
                display_warn = False
                for idx, cmd in cmds[1:]:
                    if cmd != cmds[0][1]:
                        if not display_warn:
                            display_warn = True
                            KR2RML.logger.warning("Table: %s. Conflict between command: \n\t+ %s \n\t+ %s", tbl.id, cmds[0][1], cmd)
                        else:
                            print("\t+", cmd)

                # only keep final commands
                for idx, cmd in cmds[:-1]:
                    delete_commands.append(idx)

                if isinstance(cmds[0][1], SetInternalLinkCmd):
                    # need to update source_uri & target_uri first (for duplicate commands, source_uri, target_uri = None)
                    key = (cmds[-1][1].source_id, cmds[-1][1].link_lbl, cmds[-1][1].target_id)
                    for idx, cmd in cmds[:-1]:
                        if (cmd.source_id, cmd.link_lbl, cmd.target_id) == key:
                            cmds[-1][1].source_uri = cmd.source_uri
                            cmds[-1][1].target_uri = cmd.target_uri
                            break

        delete_commands.sort(reverse=True)
        for idx in delete_commands:
            commands.pop(idx)

        super().__init__(commands)

    def to_yaml(self, fpath: Path):
        with open(fpath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=4)


if __name__ == '__main__':
    path = '/home/rook/workspace/DataIntegration/SourceModeling/data/mohsen-data/museum/cleaned-edm/sources/s04-ima-artworks.xml'
    kr2rml_file = '/home/rook/workspace/DataIntegration/SourceModeling/data/mohsen-data/museum/edm/models-r2rml/s04-ima-artworks-model.ttl'
    dataset = "museum_edm"

    tbl = DataTable.load_from_file(Path(path))
    # print(tbl.head(5).to_string())
    transformer = KR2RML(get_ontology(dataset), tbl, Path(kr2rml_file))

    transformer.apply_build(tbl)
    print(tbl.head(5).to_string())
    transformer.to_yaml("test.yml")
