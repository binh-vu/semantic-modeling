from collections import defaultdict
from itertools import chain

from fuzzywuzzy import process
from prompt_toolkit.completion import Completer, Completion
from typing import List

from semantic_modeling.data_io import get_ontology
from semantic_modeling.utilities.ontology import Ontology


class StringCompleter(Completer):

    def __init__(self, choices: List[str]) -> None:
        self.choices = choices
        self.extended_choices = []

    @staticmethod
    def get_predicate_completer(ont: Ontology) -> 'StringCompleter':
        return StringCompleter(ont.get_predicates())

    def get_completions(self, document, complete_event):
        results = process.extract(document.text, self.choices, limit=10)
        for result in results:
            yield Completion(result[0], start_position=-len(document.text))


class ClassCompleter(Completer):

    def __init__(self, ont: Ontology) -> None:
        self.choices = ont.get_classes()
        self.extended_choices = []

    def add_node_id(self, node_id: str) -> None:
        if node_id not in self.extended_choices:
            self.extended_choices.append(node_id)

    def get_completions(self, document, complete_event):
        limit = 5
        res = process.extract(document.text, self.choices, limit=limit)
        res2 = process.extract(document.text, self.extended_choices, limit=limit * 2)

        old_classes = {c[:-1]: s for c, s in res2}
        new_classes = [(c + "1 (add)", s) for c, s in res if c not in old_classes]

        old_classes_counter = defaultdict(lambda : 0)
        for c, s in res2:
            old_classes_counter[c[:-1]] = max(old_classes_counter[c[:-1]], int(c[-1]))

        for c, n in old_classes_counter.items():
            new_classes.append((f"{c}{n+1} (add)", old_classes[c]))

        choices = sorted(new_classes + res2, reverse=True, key=lambda x: x[1])
        for result in choices:
            yield Completion(result[0], start_position=-len(document.text))