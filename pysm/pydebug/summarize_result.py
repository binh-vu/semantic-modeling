#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas
import sys
from typing import *

from semantic_modeling.utilities.serializable import deserializeCSV

if __name__ == '__main__':
    files = sys.argv[1:]
    # print("Summarize files: ", "\n".join(files))

    header = None
    rows = []

    for file in files:
        eval = deserializeCSV(file)
        assert eval[0][0] == 'source' and eval[-1][0] == 'average'

        if header is None:
            header = eval[0][1:]
        else:
            assert header == eval[0][1:]
        rows.append([float(x) for x in eval[-1][1:]])

    df = pandas.DataFrame(data=rows, columns=header)
    # print(df.head())
    # print(df.mean(axis=0))
    print('F1:', df.mean(axis=0)['f1'])