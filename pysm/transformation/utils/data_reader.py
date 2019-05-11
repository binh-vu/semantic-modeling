#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import ujson
import xml
from pathlib import Path
from typing import Dict, Tuple, List, Set, Union, Optional

import chardet
import xmltodict


def load_xml(fpath: Path) -> List[dict]:
    """Load data from XML file"""
    with open(fpath, 'rb') as f:
        content = f.read().decode("utf-8")
        try:
            data = xmltodict.parse(content)
        except xml.parsers.expat.ExpatError as e:
            content = content.split('\n')
            content.append('</root>')
            for i in range(len(content)):
                if not content[i].strip().startswith('<?'):
                    content[i:i+1] = ['<root>', content[i]]
                    break

            content = '\n'.join(content)

            data = xmltodict.parse(content)
            data = data['root']

    return data


def load_json(fpath: Path, json_line: str='auto') -> List:
    """
        Load data from JSON file
        :param fpath: path to file, such as: museum/test.csv
        :param json_line: auto detect whether json file is JSON Line or not
        :return:
    """
    try:
        with open(fpath, 'rb') as f:
            data = ujson.load(f)
    except ValueError as e:
        data = []
        with open(fpath, 'rb') as f:
            for row in f:
                data.append(ujson.loads(row))

    return data


def load_csv(fpath: Path, header=True) -> List[str]:
    """
        Load data from CSV file
        :param fpath: path to file, such as: museum/test.csv
        :param header: whether use the first line as header or not
        :return:
    """
    data = []
    with open(fpath, 'rb') as f:
        content = f.read()

    try:
        content = content.decode("utf-8")
    except UnicodeDecodeError:
        encoding = chardet.detect(content)['encoding']
        content = content.decode(encoding)
    lines = content.splitlines()

    reader = csv.reader([s + "\n" for s in lines])
    if header:
        header = [s.strip() for s in next(reader)]
        for row in reader:
            data.append(dict(zip(header, row)))
    else:
        for row in reader:
            data.append(row)

    return data