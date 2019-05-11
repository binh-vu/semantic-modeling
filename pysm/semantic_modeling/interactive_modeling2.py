#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import *


def print_link(source_id: str, target_id: str, lbl: str):
    print(f"""-   _type_: SetInternalLink
    source_id: {source_id}
    source_uri: {source_id[:-1]}
    target_id: {target_id}
    target_uri: {target_id[:-1]}
    link_lbl: {lbl}""")


class_uri = None
predicate_uri = None


def expand_node(node_id: str) -> str:
    global class_uri
    if class_uri is None:
        class_uri = {
            "obj": "crm:E22_Man-Made_Object",
            "prod": "crm:E12_Production",
            "actor": "crm:E39_Actor",
            "appell": "crm:E82_Actor_Appellation",
            "begin": "crm:E63_Beginning_of_Existence",
            "end": "crm:E64_End_of_Existence",
            "time": "crm:E52_Time-Span",
            "group": "crm:E74_Group",
            "id": "crm:E42_Identifier",
            "title": "crm:E35_Title",
            "typeass": "crm:E17_Type_Assignment",
            "type": "crm:E55_Type",
            "ling": "crm:E33_Linguistic_Object",
            "owner": 'crm:E40_Legal_Body',
            "doc": "foaf:Document",
            "img": "crm:E38_Image",
            "concept": "skos:Concept",
        }

    return class_uri[node_id[:-1]] + node_id[-1]


def expand_edge(edge: str) -> str:
    global predicate_uri
    if predicate_uri is None:
        predicate_uri = {
            "subj": "dcterms:subject",
            "hasimg": "crm:P138i_has_representation",
            "homepage": "foaf:homepage",
            "tech": "crm:P32_used_general_technique",
            'owner': 'crm:P52_has_current_owner',
            "refer": "crm:P67i_is_referred_to_by",
            "class": "crm:P41i_was_classified_by",
            "ass": "crm:P42_assigned",
            "title": "crm:P102_has_title",
            "id": "crm:P1_is_identified_by",
            "prod": "crm:P108i_was_produced_by",
            "time": "crm:P4_has_time-span",
            "born": "crm:P92i_was_brought_into_existence_by",
            "die": "crm:P93i_was_taken_out_of_existence_by",
            "name": "crm:P131_is_identified_by",
            "nation": "crm:P107i_is_current_or_former_member_of",
            "madeby": "crm:P14_carried_out_by",
            "val": "rdf:value",
            "lbl": "rdfs:label",
            "karmaid": "karma:classLink"
        }

    return predicate_uri[edge]


if __name__ == '__main__':
    while True:
        link = input("Enter: ")
        source_id, lbl, target_id = link.split("-")
        print_link(expand_node(source_id), expand_node(target_id), expand_edge(lbl))
