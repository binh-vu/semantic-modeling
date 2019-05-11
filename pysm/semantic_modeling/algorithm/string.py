#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
from typing import Dict, Tuple, List, Set, Union, Optional


def auto_wrap(word: str, max_char_per_line: int, delimiters: List[str] = None, camelcase_split: bool = True) -> str:
    """
    Treat this as optimization problem, where we trying to minimize the number of line break
    but also maximize the readability in each line, i.e: maximize the number of characters in each lines

    Using greedy search.
    :param word:
    :param max_char_per_line:
    :param delimiters:
    :return:
    """
    # split original word by the delimiters
    if delimiters is None:
        delimiters = [' ', ':', '_', '/']

    sublines: List[str] = [""]
    for i, c in enumerate(word):
        if c not in delimiters:
            sublines[-1] += c

            if camelcase_split and not c.isupper() and i + 1 < len(word) and word[i + 1].isupper():
                # camelcase_split
                sublines.append("")
        else:
            sublines[-1] += c
            sublines.append("")

    new_sublines: List[str] = [""]
    for line in sublines:
        if len(new_sublines[-1]) + len(line) <= max_char_per_line:
            new_sublines[-1] += line
        else:
            new_sublines.append(line)

    return "\n".join(new_sublines)


camel_reg = re.compile('.+?(?:(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z0-9])|$)')
split_reg = re.compile("_| |:|\.")


def tokenize_label(lbl: str) -> List[str]:
    result = []
    for name in split_reg.split(lbl):
        for match in camel_reg.finditer(name):
            result.append(match.group(0))

    return result


def align_table(rows: List[List[str]], align: Union[str, List[str]]='left'):
    col_widths: List[int] = [0] * len(rows[0])
    # compute col widths
    for row in rows:
        for i, col in enumerate(row):
            col_widths[i] = max(col_widths[i], len(col))

    if isinstance(align, str):
        align = [align] * len(col_widths)

    new_rows = []
    for row in rows:
        new_row = []
        for i, col in enumerate(row):
            if align[i] == "left":
                new_col = f"%-{col_widths[i]}s" % col
            elif align[i] == 'right':
                new_col = f"%{col_widths[i]}s" % col
            elif align[i] == "center":
                new_col = col.center(col_widths[i])
            else:
                assert False
            new_row.append(new_col)

        new_rows.append(new_row)
    return new_rows

