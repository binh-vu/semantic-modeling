#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Define helper for Jupyter Notebook"""

import json2html
from IPython.display import display, HTML


def print_dict(dict_object):
    display(HTML(json2html.json2html.convert(json=dict_object)))


def percentage(a: float, b: float) -> str:
    return '%.2f%% (%s/%s)' % (a * 100.0 / b, a, b)
