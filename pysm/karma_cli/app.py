import shutil
import time
from pathlib import Path
from typing import *
from prompt_toolkit import prompt, HTML, print_formatted_text, PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.key_binding import KeyBindings
from fuzzywuzzy import process

from karma_cli.app_misc import CliSemanticType, NodeIdStr
from karma_cli.assistant.link_suggestion import LinkSuggestion
from karma_cli.assistant.semantic_typing import SemanticTyping
from karma_cli.completer import StringCompleter, ClassCompleter
from karma_cli.worksheet import Worksheet
from semantic_modeling.config import config
from semantic_modeling.data_io import get_ontology, get_semantic_models
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeYAML
from transformation.models.data_table import DataTable
from transformation.r2rml.commands.modeling import SetSemanticTypeCmd, SetInternalLinkCmd
from transformation.r2rml.commands.pytransform import PyTransformNewColumnCmd
from transformation.r2rml.r2rml import R2RML, Command


class Menu:
    LVL_0_BUILD_MODEL = "lvl_0_build_model"
    LVL_0_EDIT_MODEL = "lvl_0_edit_model"
    LVL_0_SAVE_MODEL = "lvl_0_save_model"

    def __init__(self, name: str):
        self.name = name
        self.is_init: bool = False


class Exec:

    def __init__(self, proceed: Callable, abort: Callable):
        self.proceed = proceed
        self.abort = abort

    @staticmethod
    def pass_func():
        pass


class App:
    instance = None

    def __init__(self, ont: Ontology, model_file: Path, tbl: DataTable, styper: SemanticTyping, link_suggestion: LinkSuggestion) -> None:
        self.ont = ont
        self.menu = []
        self.model_file = model_file
        self.tbl = tbl
        self.exec_buffer: List[Exec] = []
        self.styper = styper
        self.link_suggestion = link_suggestion
        self.class_completer = ClassCompleter(ont)
        self.predicate_completer = StringCompleter.get_predicate_completer(ont)

        if model_file.exists():
            self.worksheet = Worksheet.from_file(model_file, tbl)
            # make a backup first
            backup_file = model_file.parent / f"{model_file.name}.backup"
            # assert not backup_file.exists(), f"Cannot run when we have backup file: {backup_file}"
            shutil.copyfile(model_file, backup_file)
        else:
            self.worksheet = Worksheet.new(tbl)

        for node_id in self.worksheet.progressive_graph.get_class_node_ids():
            self.class_completer.add_node_id(node_id)

        self.bindings = KeyBindings()
        self.bindings.add('c-p')(self.render_sm)
        self.bindings.add('c-u')(self.undo_last_cmd)
        self.bindings.add('c-r')(self.redo_last_cmd)
        self.bindings.add('c-s')(self.save)
        self.last_cmd = None

    def start(self):
        assert len(self.exec_buffer) == 0, "Cannot call start twice"
        session = PromptSession(bottom_toolbar=self.app_bottom_toolbar(), key_bindings=self.bindings)

        while True:
            try:
                # STEP 1: get next command from MENU first
                if len(self.menu) == 0:
                    input_str = HTML(
                        '[MENU-0] Enter your choice (m<p fg="#c0c0c0">odel</p> / e<p fg="#c0c0c0">dit</p>): ')
                else:
                    menu = self.menu[-1]
                    if menu.name == Menu.LVL_0_BUILD_MODEL:
                        if not menu.is_init:
                            menu.attrs = [attr for attr in self.worksheet.history.all_attrs if not (self.worksheet.history.is_attr_ignore(attr) or self.worksheet.history.is_attr_processed(attr))]
                            menu.pivot = 0
                            menu.old_pivot = 0
                            menu.is_init = True

                        for i in range(menu.pivot, len(menu.attrs)):
                            attr = menu.attrs[i]
                            if self.worksheet.history.is_attr_ignore(attr) or self.worksheet.history.is_attr_processed(attr):
                                continue

                            input_str = HTML(f"[MENU-1] Current attribute: <b>{attr}</b>. Enter your choice ("
                                             f"<b>s</b><p fg='#c0c0c0'>kip</p> / "
                                             f"<b>i</b><p fg='#c0c0c0'>gnore</p> / "
                                             f'<p fg="#c0c0c0">set internal </p><b>l</b><p fg="#c0c0c0">ink</p> / '
                                             f'<b>pyt</b><p fg="#c0c0c0">ransform</p> / '
                                             f"<p fg='#c0c0c0'>set semantic </p><b>t</b><p fg='#c0c0c0'>ype</p>): ")

                            menu.pivot = i
                            break
                        else:
                            menu.pivot = len(menu.attrs)
                            input_str = HTML(f'[MENU-1] No attributes left. Enter your choice ('
                                             f'<p fg="#c0c0c0">set internal </p><b>l</b><p fg="#c0c0c0">ink</p> / '
                                             f'<b>pyt</b><p fg="#c0c0c0">ransform</p>): ')
                    elif menu.name == Menu.LVL_0_EDIT_MODEL:
                        input_str = HTML(f'[MENU-2] Enter your choice ('
                                         f'<b>d</b><p fg="#c0c0c0">elete link</p> / '
                                         f'<b>u</b><p fg="#c0c0c0">pdate link</p>): ')
                    else:
                        raise NotImplementedError()

                cmd = session.prompt(input_str).strip()

                # STEP 2: apply previous change if user haven't undo it.
                if len(self.exec_buffer) > 0:
                    self.proceed_cmd()

                # STEP 3: execute next command and store the change into buffer
                if len(self.menu) == 0:
                    if cmd == "m":
                        # modeling
                        self.menu.append(Menu(Menu.LVL_0_BUILD_MODEL))
                    elif cmd == "e":
                        # edit previous approach
                        self.menu.append(Menu(Menu.LVL_0_EDIT_MODEL))
                    else:
                        print(f"Invalid cmd: `{cmd}`")
                else:
                    menu = self.menu[-1]
                    if menu.name == Menu.LVL_0_BUILD_MODEL:
                        # modeling, loop through each attribute and allow user to select their options
                        if cmd == "t":
                            # set semantic type, get some suggestion and allow user to select or add their own semantic type
                            suggestions = self.styper.suggest(menu.attrs[menu.pivot])
                            for i, stype in enumerate(suggestions):
                                print_formatted_text(HTML(
                                    f"[{i}.] score = {stype.score:.5f} | node_id: {stype.node_id_str} | predicate: {stype.predicate}"))

                            if len(suggestions) > 0:
                                sugg_str = f"0-{len(suggestions)-1} "
                            else:
                                sugg_str = f""

                            if len(suggestions) > 0:
                                node_id = session.prompt(
                                    HTML(f'Enter your choice ({sugg_str}or node id): '), complete_while_typing=True,
                                                         completer=self.class_completer)
                            else:
                                node_id = session.prompt("Node ID: ", complete_while_typing=True,
                                                         completer=self.class_completer)

                            if node_id.isdigit() and int(node_id) < len(suggestions):
                                stype = suggestions[int(node_id)]
                            else:
                                type = session.prompt("Type: ", complete_while_typing=True,
                                                      completer=self.predicate_completer)
                                stype = CliSemanticType(NodeIdStr(node_id), type)

                            def get_proceed(class_completer: ClassCompleter, worksheet: Worksheet, menu: str, attr: str, next_pivot: int, stype: CliSemanticType):
                                def proceed():
                                    worksheet.add_command(
                                        SetSemanticTypeCmd(attr, stype.get_domain(), stype.predicate,
                                                           stype.get_node_id()))
                                    class_completer.add_node_id(stype.get_node_id())
                                    menu.old_pivot = next_pivot

                                return proceed

                            def get_abort(menu):
                                def abort():
                                    menu.pivot = menu.old_pivot

                                return abort

                            self.exec_buffer.append(Exec(
                                get_proceed(self.class_completer, self.worksheet, menu, menu.attrs[menu.pivot], menu.pivot + 1, stype),
                                get_abort(menu)))

                            # move menu pivot by one
                            menu.pivot += 1
                        elif cmd == "s":
                            # skip it, and push it into last queue
                            menu.attrs.append(menu.attrs[menu.pivot])
                            menu.pivot += 1

                            def get_proceed(menu, next_pivot: int):
                                def proceed():
                                    menu.old_pivot = next_pivot
                                return proceed

                            def get_abort(menu):
                                def abort():
                                    menu.pivot = menu.old_pivot
                                    menu.attrs.pop()

                                return abort

                            self.exec_buffer.append(Exec(get_proceed(menu, menu.pivot), get_abort(menu)))
                        elif cmd == "i":
                            # ignore this attributes
                            self.worksheet.history.ignore_attr(menu.attrs[menu.pivot])

                            def get_abort(worksheet: Worksheet, attr: str):
                                def abort():
                                    worksheet.history.unignore_attr(attr)

                                return abort

                            self.exec_buffer.append(Exec(Exec.pass_func, get_abort(self.worksheet, menu.attrs[menu.pivot])))
                        elif cmd == "pyt":
                            # pytransform
                            pycode = session.prompt("Enter code: > ", multiline=True).strip()
                            new_attr = session.prompt("New attributes: ").strip()

                            input_attrs = session.prompt("Input attributes (comma separated): ", complete_while_typing=True, completer=StringCompleter(self.worksheet.history.all_attrs)).strip()
                            input_attrs = input_attrs.split(",")
                            command = PyTransformNewColumnCmd(input_attrs, new_attr, pycode, "")

                            def get_proceed(worksheet: Worksheet, cmd: Command):
                                def proceed():
                                    worksheet.add_command(cmd)

                                return proceed

                            self.exec_buffer.append(Exec(get_proceed(self.worksheet, command), Exec.pass_func))
                        elif cmd == "l":
                            # set internal link
                            target_id = session.prompt("Enter target node: ", complete_while_typing=True, completer=StringCompleter(self.worksheet.progressive_graph.get_class_node_ids()))
                            target_id = NodeIdStr(target_id)
                            suggestions = self.link_suggestion.suggest_incoming_link(target_id.get_node_id(), self.worksheet.progressive_graph)

                            for i, (node_id_str, predicate) in enumerate(suggestions):
                                print(f"[{i}]. Class: {node_id_str.node_id_str}. Predicate: {predicate}")

                            if len(suggestions) > 0:
                                sugg_str = f"0-{len(suggestions)-1} "
                            else:
                                sugg_str = f""

                            source_id = session.prompt(f"Enter your choice ({sugg_str}or class node): ", complete_while_typing=True,
                                                       completer=self.class_completer)
                            if source_id.isdigit() and int(source_id) < len(suggestions):
                                source_id, predicate = suggestions[int(source_id)]
                            else:
                                source_id = NodeIdStr(source_id)
                                predicate = session.prompt(f"Enter predicate: ", complete_while_typing=True, completer=self.predicate_completer)

                            command = SetInternalLinkCmd(source_id.get_node_id(), target_id.get_node_id(), predicate, source_id.get_domain(), target_id.get_domain())

                            def get_proceed(worksheet: Worksheet, class_completer: ClassCompleter, source_id: NodeIdStr, cmd: Command):
                                def proceed():
                                    worksheet.add_command(cmd)
                                    if source_id.is_new():
                                        class_completer.add_node_id(source_id.get_node_id())

                                return proceed

                            self.exec_buffer.append(Exec(get_proceed(self.worksheet, self.class_completer, source_id, command), Exec.pass_func))
                        else:
                            print(f"Invalid cmd: `{cmd}`")
                    elif menu.name == Menu.LVL_0_EDIT_MODEL:
                        if cmd == "d":
                            source_id = session.prompt("Enter source id: ", complete_while_typing=True, completer=self.class_completer)
                            target_id = session.prompt("Enter target id: ", complete_while_typing=True, completer=self.class_completer)
                            predicate = session.prompt("Enter predicate: ", complete_while_typing=True, completer=self.predicate_completer)

                            try:
                                source = self.worksheet.progressive_graph.get_node_by_id(source_id)
                                target = self.worksheet.progressive_graph.get_node_by_id(target_id)
                                edge = [e for e in target.incoming_edges if e.label == predicate][0]

                                assert edge.source_node == source
                            except Exception as e:
                                print("Error while locating exact edge..", e)
                        elif cmd == "u":
                            pass
                        else:
                            print(f"Invalid cmd: `{cmd}`")
                    else:
                        raise NotImplementedError()
            except KeyboardInterrupt:
                if len(self.menu) == 0:
                    # stop application
                    break
                else:
                    # go up one level
                    if len(self.exec_buffer) > 0:
                        self.proceed_cmd()
                    self.menu.pop()
            except EOFError:
                print_formatted_text(HTML("STOP CLI APP!!"))
                # stop application immediately
                break

    def undo_last_cmd(self, event):
        def noti():
            print("Abort last cmd")

        if len(self.exec_buffer) > 0:
            self.last_cmd = self.exec_buffer.pop()
            self.last_cmd.abort()
            run_in_terminal(noti)

    def redo_last_cmd(self, event):
        def noti():
            print("Redo last cmd")

        if self.last_cmd is not None:
            self.exec_buffer.append(self.last_cmd)
            self.proceed_cmd()
            run_in_terminal(noti)

    def proceed_cmd(self):
        self.exec_buffer.pop().proceed()
        self.last_cmd = None

    def render_sm(self, event):
        def noti():
            print("Render graph")

        self.worksheet.get_graph().render2pdf("/tmp/app.pdf")
        run_in_terminal(noti)

    def save(self, event):
        def noti():
            print("Saved!!")

        if len(self.exec_buffer) > 0:
            self.proceed_cmd()
        self.worksheet.save(self.model_file)
        run_in_terminal(noti)

    def app_bottom_toolbar(self):
        return HTML('Tips: ctrl-p (render page), ctrl-u (undo), ctrl-r (redo), ctrl-s (save), ctrl-c (abort current prompt)')


if __name__ == '__main__':
    dataset = "museum_crm"
    sm_names = [sm.id for sm in get_semantic_models("museum_edm")]
    ont = get_ontology(dataset)
    train_sms = get_semantic_models(dataset)
    R2RML.load_python_scripts(Path(config.datasets[dataset].python_code.as_path()))

    dataset_dir = Path("/workspace/semantic-modeling/data/museum-jws-crm")
    data_files = []
    # data_files = [file for file in (dataset_dir / "tmp").iterdir() if file.name.startswith("s")]
    for file in (dataset_dir / "sources").iterdir():
        if file.name.startswith("s"):
            data_files.append(file)

    for sm_name in sm_names:
        # if int(sm_name[1:3]) <= 18:
        #     continue
        if not sm_name.startswith("s28"):
            continue

        data_file = [file for file in data_files if file.name.startswith(sm_name)][0]
        print("Model source", sm_name)
        tbl = DataTable.load_from_file(data_file)
        app = App(ont, Path(f"/workspace/semantic-modeling/data/museum-jws-crm/models-y2rml/{sm_name}-model.yml"), tbl, SemanticTyping(), LinkSuggestion(train_sms))
        app.start()

        break
