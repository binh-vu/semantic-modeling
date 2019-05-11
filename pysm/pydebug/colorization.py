#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *
from data_structure import *

class WrappedLink(GraphLink):

    # noinspection PyMissingConstructor
    def __init__(self, is_correct: bool) -> None:
        self.is_correct = is_correct

    def get_dot_format(self, max_text_width: int):
        label = self.get_printed_label(max_text_width).encode('unicode_escape').decode()
        if self.is_correct:
            return '"%s" -> "%s"[color="black",fontcolor="black",label="%s"];' % (self.source_id, self.target_id, label)
        return '"%s" -> "%s"[color="red",penwidth=2.5,fontcolor="black",label="%s"];' % (self.source_id, self.target_id,
                                                                                         label)


class WrappedOutputLink(GraphLink):

    # noinspection PyMissingConstructor
    def __init__(self, gold_label: bool, pred_label: bool) -> None:
        self.pred_label = pred_label
        self.gold_label = gold_label

    def get_dot_format(self, max_text_width: int):
        label = self.get_printed_label(max_text_width).encode('unicode_escape').decode()
        # color meaning
        # - black: mean it is true label, and you predict it right (nothing special)
        # - green: mean it is false label, and you predict it right (kind of rewarding)
        # - dark orange: mean it is true label, and you predict it wrong (kind of alarm)
        # - red: mean it is false label, and you predict it wrong (error, should pay more attention)
        if self.gold_label and self.pred_label:
            return '"%s" -> "%s"[color="black",fontcolor="black",label="%s"];' % (self.source_id, self.target_id, label)
        if not self.gold_label and not self.pred_label:
            return '"%s" -> "%s"[color="darkgreen",penwidth=2.5,fontcolor="black",label="%s"];' % (self.source_id,
                                                                                                   self.target_id,
                                                                                                   label)
        if self.gold_label and not self.pred_label:
            return '"%s" -> "%s"[color="darkorange2",penwidth=2.5,fontcolor="black",label="%s"];' % (self.source_id,
                                                                                                     self.target_id,
                                                                                                     label)
        assert not self.gold_label and self.pred_label
        return '"%s" -> "%s"[color="red4",penwidth=2.5,fontcolor="black",label="%s"];' % (self.source_id,
                                                                                          self.target_id, label)


def colorize_prediction(pred_sm: Graph, link2label: Dict[int, bool], pred_link2label: Dict[int, bool]=None, name: str=None):
    if name is None:
        name = pred_sm.name
    else:
        name = name.encode("utf-8")

    g = Graph(name=name)

    for n in pred_sm.iter_nodes():
        g.add_new_node(n.type, n.label)
    for e in pred_sm.iter_links():
        if pred_link2label is None:
            e_prime = WrappedLink(link2label[e.id])
        else:
            e_prime = WrappedOutputLink(link2label[e.id], pred_link2label[e.id])

        g.real_add_new_link(e_prime, e.type, e.label, e.source_id, e.target_id)
    return g