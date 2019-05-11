from pathlib import Path
from typing import *

from data_structure import Graph, GraphNode, GraphNodeType
from karma_cli.app_misc import CliGraph
from semantic_modeling.utilities.serializable import deserializeYAML, serializeYAML
from transformation.models.data_table import DataTable
from transformation.models.table_schema import Schema
from transformation.r2rml.commands.alter_structure import AddLiteralColumnCmd, ZipAttributesCmd, UnpackOneElementListCmd
from transformation.r2rml.commands.modeling import SetSemanticTypeCmd, SetInternalLinkCmd
from transformation.r2rml.commands.pytransform import PyTransformNewColumnCmd
from transformation.r2rml.r2rml import Command, R2RML


class Worksheet:
    def __init__(self, tbl: DataTable, history: 'History', commands: List[Command]):
        self.tbl = tbl
        self.commands = commands
        self.history = history
        self.progressive_table = tbl.head(100).clone()
        self.progressive_table_cmds = set()
        self.progressive_graph: CliGraph = CliGraph()

        for i, cmd in enumerate(commands):
            self.apply_command(i, cmd)

    @staticmethod
    def new(tbl: DataTable) -> 'Worksheet':
        return Worksheet(tbl, History(tbl.schema.get_attr_paths(), tbl.schema.get_attr_paths(), [], []), [])

    @staticmethod
    def from_file(model_file: Path, tbl: DataTable) -> 'Worksheet':
        assert model_file.exists() and model_file.suffix in {".yml", ".yaml"}

        conf = deserializeYAML(model_file)
        r2rml = R2RML.load_from_file(model_file)
        commands = r2rml.commands

        if 'history' not in conf:
            small_tbl = tbl.head(10).clone()
            # apply commands to get new attributes
            r2rml.apply_cmds(small_tbl)

            history = History(
                input_attrs=tbl.schema.get_attr_paths(),
                all_attrs=small_tbl.schema.get_attr_paths(),
                ignored_attrs=[],
                processed_attrs=[])
        else:
            history = History(conf['history']['input_attrs'], conf['history']['all_attrs'],
                              conf['history']['ignored_attrs'], conf['history']['processed_attrs'])

        return Worksheet(tbl, history, commands)

    def save(self, model_file: Path) -> None:
        commands = []
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, AddLiteralColumnCmd))
        commands.extend(cmd.to_dict() for cmd in self.commands
                        if isinstance(cmd, (PyTransformNewColumnCmd, ZipAttributesCmd, UnpackOneElementListCmd)))
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, SetSemanticTypeCmd))
        commands.extend(cmd.to_dict() for cmd in self.commands if isinstance(cmd, SetInternalLinkCmd))

        serializeYAML({"commands": commands, 'history': self.history.to_dict()}, model_file)

    def add_command(self, cmd: Command):
        self.commands.append(cmd)
        self.apply_command(len(self.commands) - 1, cmd, exec_pytransform=False)

    def apply_command(self, cmd_idx: int, cmd: Command, exec_pytransform: bool=True):
        if isinstance(cmd, PyTransformNewColumnCmd):
            self.progressive_table_cmds.add(cmd_idx)
            new_attr_path = Schema.PATH_DELIMITER.join(
                cmd.input_attr_paths[0].split(Schema.PATH_DELIMITER)[:-1] + [cmd.new_attr_name])
            # assert not tbl.schema.has_attr_path(new_attr_path)
            # TODO: fix me!! not handle list of input attr path properly (cmd.input_attr_paths[0])
            self.progressive_table.schema.add_new_attr_path(new_attr_path,
                                                            self.progressive_table.schema.get_attr_type(
                                                                cmd.input_attr_paths[0]), cmd.input_attr_paths[-1])
            if exec_pytransform:
                R2RML.pytransform(self.progressive_table, cmd)
            if new_attr_path not in self.history.all_attrs:
                self.history.all_attrs.append(new_attr_path)
        elif isinstance(cmd, SetSemanticTypeCmd):
            self.progressive_graph.add_node(cmd.node_id)
            self.progressive_graph.add_data_node(cmd.input_attr_path)
            self.progressive_graph.add_edge(cmd.type, cmd.node_id, cmd.input_attr_path)
            self.history.mark_attr_as_processed(cmd.input_attr_path)
        elif isinstance(cmd, SetInternalLinkCmd):
            self.progressive_graph.add_node(cmd.source_id)
            self.progressive_graph.add_node(cmd.target_id)
            self.progressive_graph.add_edge(cmd.link_lbl, cmd.source_id, cmd.target_id)
        elif isinstance(cmd, ZipAttributesCmd):
            for row in self.progressive_table.rows:
                cmd.zip_attributes(row)
            # TODO: fix me!! re-build schema, which is very expensive
            self.progressive_table.rebuild_schema()
            new_attrs = set(self.progressive_table.schema.get_attr_paths())
            self.history.all_attrs = [attr for attr in self.history.all_attrs if attr in new_attrs]
            for attr in new_attrs:
                if attr not in self.history.all_attrs:
                    self.history.all_attrs.append(attr)

    def get_graph(self) -> Graph:
        commands = [cmd for i, cmd in enumerate(self.commands) if i not in self.progressive_table_cmds]
        sm = R2RML(commands).apply_cmds(self.progressive_table)
        for attr in self.history.iter_unignore_attr():
            if not sm.has_attr(attr):
                sm.graph.add_new_node(GraphNodeType.DATA_NODE, attr.encode())
        return sm.graph


class History:
    def __init__(self, input_attrs: List[str], all_attrs: List[str], ignored_attrs: List[str],
                 processed_attrs: List[str]):
        self.input_attrs = input_attrs
        self.all_attrs = all_attrs
        self.ignored_attrs = ignored_attrs
        self.processed_attrs = processed_attrs
        self._ignored_attrs = set(ignored_attrs)
        self._processed_attrs = set(processed_attrs)

    def iter_unignore_attr(self):
        for attr in self.all_attrs:
            if attr not in self._ignored_attrs:
                yield attr

    def is_attr_ignore(self, attr: str) -> bool:
        return attr in self._ignored_attrs

    def is_attr_processed(self, attr: str) -> bool:
        return attr in self._processed_attrs

    def mark_attr_as_processed(self, attr: str) -> None:
        if attr not in self._processed_attrs:
            self.processed_attrs.append(attr)
            self._processed_attrs.add(attr)

    def ignore_attr(self, attr: str) -> None:
        if attr not in self._ignored_attrs:
            self.ignored_attrs.append(attr)
            self._ignored_attrs.add(attr)

    def unignore_attr(self, attr: str) -> None:
        self.ignored_attrs = [a for a in self.ignored_attrs if attr != a]
        self._ignored_attrs.remove(attr)

    def to_dict(self) -> dict:
        return {
            "input_attrs": self.input_attrs,
            "all_attrs": self.all_attrs,
            "ignored_attrs": self.ignored_attrs,
            "processed_attrs": self.processed_attrs
        }
