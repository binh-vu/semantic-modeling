#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import subprocess
import uuid
from pathlib import Path
from typing import Callable, Any, Type, Optional, List, Set, TypeVar, TYPE_CHECKING, Union

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from data_structure.graph_c.graph import GraphNode, GraphLink, Graph

V = TypeVar('V', bound='GraphNode')
E = TypeVar('E', bound='GraphLink')
G = TypeVar('G', bound='Graph[V, E]')


# noinspection PyUnusedLocal
def _empty_dict(x: Any) -> dict:
    return {}


def graph2dict(graph: G,
               graph2meta: Optional[Callable[['Graph'], dict]]=None,
               node2meta: Optional[Callable[[V], dict]]=None,
               link2meta: Optional[Callable[[E], dict]]=None) -> dict:
    """Serialize graph to dictionary object that can be dumped to JSON file"""

    graph2meta = graph2meta or _empty_dict
    node2meta = node2meta or _empty_dict
    link2meta = link2meta or _empty_dict

    obj: dict = {
        "name": graph.name,
        "index_node_type": graph.index_node_type,
        "index_node_label": graph.index_node_label,
        "index_link_label": graph.index_link_label,
        "nodes": [],
        "links": [],
        **graph2meta(graph)
    }

    for node in graph.iter_nodes():
        obj["nodes"].append({"id": node.id, "type": node.type, "label": node.label, **node2meta(node)})
    for link in graph.iter_links():
        obj["links"].append({
            "id": link.id,
            "type": link.type,
            "label": link.label,
            "source_id": link.source_id,
            "target_id": link.target_id,
            **link2meta(link)
        })

    return obj


def dict2graph(obj: dict,
               GraphClass: Type[G],
               NodeClass: Type[V],
               LinkClass: Type[E],
               gmeta2args: Optional[Callable[[dict], dict]]=None,
               nmeta2args: Optional[Callable[[dict], dict]]=None,
               emeta2args: Optional[Callable[[dict], dict]]=None) -> G:
    """Deserialize a dictionary object to a graph"""
    gmeta2args = gmeta2args or _empty_dict
    nmeta2args = nmeta2args or _empty_dict
    emeta2args = emeta2args or _empty_dict

    # noinspection PyCallingNonCallable
    g = GraphClass(
        index_node_type=obj['index_node_type'],
        index_node_label=obj['index_node_label'],
        index_link_label=obj['index_link_label'],
        estimated_n_nodes=len(obj['nodes']),
        estimated_n_links=len(obj['links']),
        name=obj['name'].encode('utf-8'),
        **gmeta2args(obj))

    for nobj in obj['nodes']:
        # noinspection PyCallingNonCallable
        node = g.real_add_new_node(NodeClass(**nmeta2args(nobj)), nobj["type"], nobj["label"].encode('utf-8'))
        assert nobj["id"] == node.id

    for eobj in obj['links']:
        # noinspection PyCallingNonCallable
        link = g.real_add_new_link(LinkClass(**emeta2args(eobj)), eobj['type'], eobj['label'].encode('utf-8'), eobj['source_id'], eobj['target_id'])
        assert eobj["id"] == link.id

    return g


def graph2dot(g: 'Graph',
              f_output: Union[str, Path],
              max_text_width: int,
              ignored_node_ids: Optional[Set[int]]=None,
              ignored_link_ids: Optional[Set[int]]=None) -> None:
    ignored_node_ids = ignored_node_ids or set()
    ignored_link_ids = ignored_link_ids or set()
    nodes: List[V] = [n for n in g.iter_nodes() if n.id not in ignored_node_ids]
    links: List[E] = [e for e in g.iter_links() if e.id not in ignored_link_ids]

    with open(f_output, 'w') as f:
        f.write('''digraph n0 {
fontcolor="blue"
remincross="true"
label="%s"
''' % g.name.decode('utf-8'))
        for n in nodes:
            f.write(n.get_dot_format(max_text_width) + "\n")

        for l in links:
            f.write(l.get_dot_format(max_text_width) + "\n")
        f.write('}\n')


def graph2img(g: 'Graph',
              f_output: Union[str, Path],
              max_text_width: int,
              ignored_node_ids: Optional[Set[int]]=None,
              ignored_link_ids: Optional[Set[int]]=None) -> None:
    f_output = str(f_output)
    graph2dot(g, f_output + ".tmp", max_text_width, ignored_node_ids, ignored_link_ids)
    subprocess.check_call(['dot', '-Tpng', f_output + '.tmp', '-o' + f_output])
    os.remove(f_output + '.tmp')


def graph2pdf(g: 'Graph',
              f_output: Union[str, Path],
              max_text_width: int,
              ignored_node_ids: Optional[Set[int]]=None,
              ignored_link_ids: Optional[Set[int]]=None) -> None:
    f_output = str(f_output)
    graph2dot(g, f_output + ".tmp", max_text_width, ignored_node_ids, ignored_link_ids)
    subprocess.check_call(['dot', '-Tpdf', f_output + '.tmp', '-o' + f_output])
    os.remove(f_output + '.tmp')


def render_graph(g: 'Graph',
                 dpi: int,
                 max_text_width: int,
                 ignored_node_ids: Optional[Set[int]]=None,
                 ignored_link_ids: Optional[Set[int]]=None) -> None:
    f_output = '/tmp/%s.png' % uuid.uuid4()
    graph2img(g, f_output, max_text_width, ignored_node_ids, ignored_link_ids)
    img = mpimg.imread(f_output)
    os.remove(f_output)
    height, width, depth = img.shape
    fig = plt.figure(figsize=(width / dpi, height / dpi))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(img, interpolation="bilinear")
    plt.show()
