from typing import *

from pathlib import Path

from data_structure import Graph, GraphNodeType, GraphLinkType
from semantic_modeling.data_io import get_ontology
from semantic_modeling.utilities.ontology import Ontology
from semantic_modeling.utilities.serializable import deserializeJSON


class SSDAttribute:

    def __init__(self, id: int, name: str) -> None:
        self.id = id
        self.name = name


class SSD:

    def __init__(self, name: str, attrs: List[SSDAttribute], graph: Graph, ont: Ontology):
        self.name = name
        self.attributes = attrs
        self.graph = graph
        self.ont = ont

    @staticmethod
    def from_file(file: Union[str, Path], ont: Ontology) -> 'SSD':
        content = deserializeJSON(file)
        return SSD.from_json(content, ont)

    @staticmethod
    def from_json(obj: dict, ont: Ontology) -> 'SSD':
        g = Graph(True, True, True)
        node2attr = {
            x['node']: x['attribute']
            for x in obj['mappings']
        }
        idmap = {}
        raw_attributes = {}
        for raw_attr in obj['attributes']:
            assert len(raw_attr['columnIds']) == 1 and raw_attr['columnIds'][0] == raw_attr['id']
            raw_attributes[raw_attr['id']] = raw_attr

        attrs = []
        for n in obj['semanticModel']['nodes']:
            if n['type'] == 'DataNode':
                node_type = GraphNodeType.DATA_NODE
                attr = raw_attributes[node2attr[n['id']]]
                n_lbl = attr['name']
                attrs.append(SSDAttribute(n['id'], n_lbl))
            else:
                node_type = GraphNodeType.CLASS_NODE
                n_lbl = n['prefix'] + n['label']
                n_lbl = ont.simplify_uri(n_lbl)

            idmap[n['id']] = g.add_new_node(node_type, n_lbl.encode()).id

        for e in obj['semanticModel']['links']:
            e_lbl = e['prefix'] + e['label']
            e_lbl = ont.simplify_uri(e_lbl)
            g.add_new_link(GraphLinkType.UNSPECIFIED, e_lbl.encode(), idmap[e['source']], idmap[e['target']])

        return SSD(obj['name'], attrs, g, ont)

    def clear_serene_footprint(self, remove_unknown: bool=True) -> 'SSD':
        g = Graph(True, True, True)
        idmap = {}

        serene_all = None
        serene_unknown = None
        for n in self.graph.iter_nodes():
            if n.label == b"serene:All":
                serene_all = n
                continue

            if n.label == b"serene:Unknown":
                serene_unknown = n
                continue

        ignore_nodes = set()
        if serene_all is not None:
            ignore_nodes.add(serene_all.id)

        if remove_unknown and serene_unknown is not None:
            ignore_nodes.add(serene_unknown.id)
            for e in self.graph.iter_links():
                if e.source_id == serene_unknown.id:
                    assert e.get_target_node().is_data_node()
                    ignore_nodes.add(e.target_id)

        if len(ignore_nodes) == 0:
            # no serene footprint to remove
            return self

        for n in self.graph.iter_nodes():
            if n.id in ignore_nodes:
                continue

            idmap[n.id] = g.add_new_node(n.type, n.label).id
        for e in self.graph.iter_links():
            if e.label == b"serene:connect":
                continue
            if remove_unknown and e.label == b"serene:unknown":
                continue
            g.add_new_link(e.type, e.label, idmap[e.source_id], idmap[e.target_id])

        self.graph = g
        return self

    def to_dict(self) -> dict:
        links = []
        for e in self.graph.iter_links():
            label, prefix = self._get_label_and_prefix(e.label.decode())
            if e.label == b"karma:classLink":
                e_type = "ClassInstanceLink"
            elif e.get_target_node().is_data_node():
                e_type = "DataPropertyLink"
            else:
                e_type = "ObjectPropertyLink"

            links.append({
                "id": e.id,
                "label": label,
                "prefix": prefix,
                "source": e.source_id,
                "target": e.target_id,
                "type": e_type,
                "status": "ForcedByUser",
            })

        nodes = []
        for n in self.graph.iter_nodes():
            if n.is_data_node():
                prefix = ""
                e = n.get_first_incoming_link()
                ns, v0 = e.get_source_node().label.decode().split(":")
                ns, v1 = e.label.decode().split(":")
                label = f"{v0}.{v1}"
            else:
                label, prefix = self._get_label_and_prefix(n.label.decode())
            nodes.append({
                "id": n.id,
                "label": label,
                "prefix": prefix,
                "status": "ForcedByUser",
                "type": "DataNode" if n.is_data_node() else "ClassNode"
            })

        ssd = {
            "dateCreated": "2017-03-21T18:29:30.849",
            "dateModified": "2017-03-21T18:29:30.849",
            "id": 1892396233,
            "name": self.name,
            "ontologies": [],
            "semanticModel": {
                "nodes": nodes,
                "links": links
            },
            "mappings": [{"attribute": attr.id, "node": attr.id} for attr in self.attributes],
            "attributes": [
                {"columnIds": [attr.id], "id": attr.id, "label": "identity", "name": attr.name, "sql": "not implemented"}
                for attr in self.attributes
            ]
        }
        return ssd

    def _get_label_and_prefix(self, lbl: str) -> Tuple[str, str]:
        ns, val = lbl.split(":")
        return (val, self.ont.namespaces[ns])

if __name__ == '__main__':
    ont = get_ontology("museum_edm")
    ont.register_namespace("serene", "http://au.csiro.data61/serene/dev#")

    ssd = SSD.from_file("/workspace/tmp/serene-python-client/tests/resources/museum_benchmark/ssd/s01-cb.csv.ssd", ont)
    # ssd = SSD.from_file("/workspace/tmp/serene-python-client/tests/resources/museum_benchmark/ssd/s02-dma.csv.ssd", ont)
    # ssd = SSD.from_file("/workspace/tmp/serene-python-client/stp/resources/octopus_result/s07-s-13.json.cor_ssd.json", ont)
    ssd.clear_serene_footprint().graph.render(80)
    ssd.to_dict()