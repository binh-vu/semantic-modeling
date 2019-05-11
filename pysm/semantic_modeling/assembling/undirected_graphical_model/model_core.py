#!/usr/bin/python
# -*- coding: utf-8 -*-
from typing import Dict, List, Optional, Tuple, Set

from pyutils.list_utils import _

from data_structure import GraphLink
from gmtk.graph_models.variables.vector_domain import GrowableBinaryVectorDomain
from gmtk.graph_models.variables.vector_variable import LabeledBooleanVectorVariable, \
    GrowableMultiContinuousVectorVariable
from semantic_labeling import create_semantic_typer
from semantic_labeling.typer import SemanticTyper
from semantic_modeling.assembling.weak_models.attribute_scope import AttributeScope
from semantic_modeling.assembling.weak_models.cardinality_matrix import CardinalityFeatures, CardinalityMatrix
from semantic_modeling.assembling.weak_models.primary_key import PrimaryKey
from semantic_modeling.assembling.weak_models.structures.duplication_tensors import DuplicationTensors
from semantic_modeling.settings import Settings
from semantic_modeling.assembling.learning.shared_models import Example
from semantic_modeling.assembling.weak_models.local_structure import LocalStructure
from semantic_modeling.assembling.weak_models.statistic import Statistic
from semantic_modeling.assembling.weak_models.node_prob import NodeProb
from semantic_modeling.assembling.weak_models.multi_val_predicate import MultiValuePredicate
from semantic_modeling.assembling.weak_models.semantic_type_assistant import get_stype_assistant_model
from semantic_modeling.data_io import get_semantic_models
from semantic_modeling.karma.semantic_model import SemanticModel


# ######################################################################################################################
# VARIABLEs

class TripleLabel(LabeledBooleanVectorVariable):
    def __init__(self, label: bool, triple: 'Triple') -> None:
        super().__init__(label)
        self.triple: Triple = triple

    def __repr__(self):
        return "TripleLabel(source=%s, link=%s, target=%s)" % (self.triple.source.label, self.triple.link.label, self.triple.target.label)


class Triple(object):
    def __init__(self, label: bool, link: GraphLink, example: Example) -> None:
        self.label: TripleLabel = TripleLabel(label, self)
        self.link: GraphLink = link
        self.example: Example = example
        self.features: TripleFeatures = None
        self.parent: Triple = None
        self.siblings: List[Triple] = []
        self.children: List[Triple] = []
        self.is_root_triple: bool = False

        self.source = self.link.get_source_node()
        self.target = self.link.get_target_node()

    def provenance(self):
        return self.example.example_id

    def __repr__(self):
        return "Triple(source=%s, link=%s, target=%s)" % (self.source.label, self.link.label, self.target.label)


class TripleFeatures(GrowableMultiContinuousVectorVariable[str]):
    def is_observed(self) -> bool:
        return True


# ######################################################################################################################
# FEATURE EXTRACTIONS


class ExampleAnnotator:
    def __init__(self,
                 dataset: str,
                 train_source_ids: List[str],
                 load_circular_dependency: bool = True,
                 training_examples: Optional[List[Example]]=None):
        """
        :param dataset:
        :param train_source_ids:
        :param top_k_semantic_types:
        :param n_sample:
        :param load_circular_dependency:
        :param training_examples: list of training examples use to build weak models, don't need it at testing time (i.e = NULL), because weak models has been built before
        """
        self.dataset = dataset
        self.source_models = {sm.id: sm for sm in get_semantic_models(dataset)}
        self.train_source_ids = set(train_source_ids)
        self.top_k_semantic_types = Settings.get_instance().semantic_labeling_top_n_stypes

        self.training_models = [self.source_models[sid] for sid in train_source_ids]
        self.typer: SemanticTyper = create_semantic_typer(dataset, self.training_models)

        self.testing_models = [
            self.source_models[sid] for sid in set(self.source_models.keys()).difference(train_source_ids)
        ]
        self.training_examples = training_examples

        # local models
        self.multival_predicate = MultiValuePredicate.get_instance(self.training_models)
        self.statistic = Statistic.get_instance(self.training_models)
        # self.data_constraint = get_data_constraint_model(dataset, self.training_models)
        self.stype_assistant = get_stype_assistant_model(dataset, self.training_models)
        self.local_structure = LocalStructure.get_instance(self.training_models)
        self.attribute_same_scope = AttributeScope.get_instance(self.dataset)
        self.duplication_tensors = DuplicationTensors.get_instance(self.training_models)

        self.primary_key: PrimaryKey = PrimaryKey.get_instance(dataset, self.training_models)
        self.cardinality = CardinalityFeatures.get_instance(dataset)

        # STEP 1: add semantic types
        self.typer.semantic_labeling(self.training_models, self.testing_models, self.top_k_semantic_types, eval_train=True)

        # STEP 2: load circular dependency like node_prob
        if load_circular_dependency:
            self.node_prob = NodeProb(self, load_classifier=True)

    def get_stype_score(self, example: Example) -> Dict[int, float]:
        """Compute stype prob. but store in a map: data node id => prob."""
        stype_score = {}
        source_desc = self.source_models[example.get_model_id()]
        for target in example.pred_sm.iter_data_nodes():
            link = target.get_first_incoming_link()
            source = link.get_source_node()
            for stype in source_desc.get_attr_by_label(target.label.decode("utf-8")).semantic_types:
                if stype.domain.encode("utf-8") == source.label and stype.type.encode("utf-8") == link.label:
                    p_link_given_so = stype.confidence_score
                    break
            else:
                p_link_given_so = None

            stype_score[target.id] = p_link_given_so
        return stype_score

    # @profile
    def annotate(self, example: Example) -> Example:
        # STEP 1: add semantic types... dont' need to do, because example must be either in train or test...
        sm_id: str = example.get_model_id()
        assert sm_id in self.source_models
        example.annotator = self

        is_train_example: bool = sm_id in self.train_source_ids
        source: SemanticModel = self.source_models[sm_id]

        # id2attrs: Dict[int, Attribute] = {attr.id: attr for attr in sources[sm_id].attrs}
        example.node2features = {}
        example.link2features = {}
        stype_score = self.get_stype_score(example)

        # add node features from node_prob weak model
        node_prob_features = self.node_prob.feature_extraction(example.pred_sm, stype_score)
        node_probs = self.node_prob.compute_prob(node_prob_features)
        for nid, prob in node_probs.items():
            example.node2features[nid] = dict(node_prob_features[nid])
            example.node2features[nid]['node_prob'] = prob

        stype_assistant = self.stype_assistant.compute_prob(sm_id, example.pred_sm)

        # add link features
        for node in example.pred_sm.iter_class_nodes():
            outgoing_links = list(node.iter_outgoing_links())
            numbered_links = numbering_link_labels(outgoing_links)

            for link in outgoing_links:
                target = link.get_target_node()
                total_stype_score = None
                delta_stype_score = None
                ratio_stype_score = None
                p_link_given_so = None
                p_triple = None
                stype_order = None
                data_constraint_features = {}

                if target.is_class_node():
                    p_link_given_so = self.statistic.p_l_given_so(
                        node.label, link.label, target.label, default=0.5)  # half half
                    p_triple = p_link_given_so * example.node2features[link.source_id][
                        'node_prob'] * example.node2features[link.target_id]['node_prob']
                else:
                    target_stypes = source.get_attr_by_label(target.label.decode("utf-8")).semantic_types
                    n_target_stypes = len(target_stypes)
                    total_stype_score = sum(stype.confidence_score for stype in target_stypes)

                    for i, stype in enumerate(target_stypes):
                        if stype.domain.encode("utf-8") == node.label and stype.type.encode("utf-8") == link.label:
                            # data node, p_link = score of semantic type
                            p_link_given_so = stype.confidence_score
                            if i == 0 and n_target_stypes > 1:
                                delta_stype_score = stype.confidence_score - target_stypes[1].confidence_score
                            else:
                                delta_stype_score = stype.confidence_score - target_stypes[0].confidence_score

                            ratio_stype_score = stype.confidence_score / target_stypes[0].confidence_score
                            stype_order = i
                            break

                    if p_link_given_so is not None:
                        p_triple = p_link_given_so * example.node2features[link.source_id]['node_prob']

                    # add data constraint
                    # if is_train_example:
                    #     # we can use link2label, because of known models
                    #     data_constraint_features = self.data_constraint.extract_feature(sm_id, example.pred_sm, target.id,
                    #                                                                     example.link2label)
                    # else:
                    #     data_constraint_features = self.data_constraint.extract_feature(sm_id, example.pred_sm, target.id)

                example.link2features[link.id] = {
                    'p_triple': p_triple,
                    'p_link_given_so': p_link_given_so,
                    'total_stype_score': total_stype_score,
                    'stype_order': stype_order,
                    'delta_stype_score': delta_stype_score,
                    'ratio_stype_score': ratio_stype_score,
                    # 'local_constraint': data_constraint_features.get("local", None),
                    # 'global_constraint': data_constraint_features.get("global", None),
                    'stype_prob': stype_assistant.get(link.id, None)
                }

                multi_val_prob = self.multival_predicate.compute_prob(link.label, numbered_links[link.id])
                if multi_val_prob is not None:
                    example.link2features[link.id]["multi_val_prob"] = multi_val_prob

        return example

    def example2vars(self, example: Example) -> List[Triple]:
        triples: Dict[int, Triple] = {}

        for link in example.pred_sm.iter_links():
            triples[link.id] = Triple(label=example.link2label[link.id], link=link, example=example)

        for link_id, triple in triples.items():
            source_node = triple.link.get_source_node()
            parent_link = source_node.get_first_incoming_link()

            if parent_link is not None:
                triples[link_id].parent = triples[parent_link.id]
                triples[parent_link.id].children.append(triples[link_id])

            for outgoing_link in source_node.iter_outgoing_links():
                if outgoing_link.id != link_id:
                    triples[link_id].siblings.append(triples[outgoing_link.id])

        # set one triple as root
        for triple in triples.values():
            if triple.parent is None:
                triple.is_root_triple = True
                break

        return list(triples.values())

    def build_triple_features(self, triple: Triple, domain: GrowableBinaryVectorDomain[str]) -> Triple:
        json_features = triple.example.link2features[triple.link.id]
        features = TripleFeatures(domain)
        triple_target = triple.target
        triple_source = triple.source
        triple_source_label = triple_source.label.decode('utf-8')

        # if 'multi_val_prob' in json_features:
        #     features += ("multi_val_prob", max(json_features['multi_val_prob'], 0.01))
        #     features += ("single_val_prob", max(1 - json_features['multi_val_prob'], 0.01))

        if json_features['p_link_given_so'] is not None:
            if triple_target.is_data_node():
                name = "(%s---%s)" % (triple_source_label, triple.link.label.decode("utf-8"))
                features += (f'{name}=True.p_semantic_type', max(json_features['p_link_given_so'], 0.01))
                features += (f'{name}=False.p_semantic_type', max(1 - json_features['p_link_given_so'], 0.01))
                # features += (f'{name}=True.delta_p_semantic_type', min(json_features['delta_stype_score'], -0.01))
                # features += (f'{name}=False.delta_p_semantic_type', max(-1 * json_features['delta_stype_score'], 0.01))
                features += (f'{name}=True.delta_p_semantic_type', max(json_features['delta_stype_score'], 0.01))
                features += (f'{name}=False.delta_p_semantic_type', max(-1 * json_features['delta_stype_score'], 0.01))

                features += (f'{name}=True.ratio_p_semantic_type', 1 / json_features['ratio_stype_score'])
                features += (f'{name}=False.ratio_p_semantic_type', json_features['ratio_stype_score'])
                # features += (f'{name}=True.norm-p_semantic_type',
                #              max(json_features['p_link_given_so'] / json_features['total_stype_score'], 0.01))
                # features += (f'{name}=False.norm-p_semantic_type',
                #              max(1 - (json_features['p_link_given_so'] / json_features['total_stype_score']), 0.01))
                features += (f'{name}-order={json_features["stype_order"]}', 1)
                features += (f'{triple_source_label}=True.p_triple', max(json_features['p_triple'], 0.01))
                features += (f'{triple_source_label}=False.p_triple', max(1 - json_features['p_triple'], 0.01))
            else:
                features += (f'{triple_source_label}=True.p_triple', max(json_features['p_triple'], 0.01))
                features += (f'{triple_source_label}=False.p_triple', max(1 - json_features['p_triple'], 0.01))

        # if json_features['local_constraint'] is not None:
        #     features += (f"class={triple_source_label}=True.local_constraint", max(json_features['local_constraint'], 0.01))
        #     features += (f"class={triple_source_label}=False.local_constraint", max(1 - json_features['local_constraint'], 0.01))
        # if json_features['global_constraint'] is not None:
        #     features += (f"class={triple_source_label}=True.global_constraint", max(json_features['global_constraint'], 0.01))
        #     features += (f"class={triple_source_label}=False.global_constraint", max(1 - json_features['global_constraint'], 0.01))

        if json_features['stype_prob'] is not None:
            features += (f"True.stype_prob", max(json_features['stype_prob'], 0.01))
            features += (f"False.stype_prob", max(1 - json_features['stype_prob'], 0.01))

        if triple.target.is_class_node() and _(triple.siblings).all(lambda t: t.target.is_class_node()):
            # if domain.has_value("source_node_no_data_child") or not domain.is_frozen:
            features += (f"class={triple_source_label}.source_node_no_data_child", 1)

        if triple.target.is_class_node() and len(triple.siblings) == 0:
            # if domain.has_value("no_siblings") or not domain.is_frozen:
            features += (f"class={triple_source_label}.no_siblings", 1)

            if triple.parent is None:
                # if domain.has_value("no_parent_&_no_siblings") or not domain.is_frozen:
                features += (f"class={triple_source_label}.no_parent_&_no_siblings", 1)

        triple.features = features
        return triple

    def build_pairwise_domain(self) -> GrowableBinaryVectorDomain[str]:
        domain = GrowableBinaryVectorDomain()
        relations = [CardinalityMatrix.ONE_TO_N, CardinalityMatrix.ONE_TO_ONE, CardinalityMatrix.N_TO_ONE, CardinalityMatrix.UNCOMPARABLE]

        for sm in self.training_models:
            for n in sm.graph.iter_class_nodes():
                if n.label not in self.primary_key:
                    continue

                primary_key = self.primary_key[n.label]
                edges = {e.label: e.get_target_node() for e in n.iter_outgoing_links()}

                if primary_key in edges:
                    # has primary key
                    for e_lbl, node in edges.items():
                        if e_lbl != primary_key:
                            feature = f"source={n.label.decode('utf-8')},x={primary_key.decode('utf-8')},y={e_lbl.decode('utf-8')},cardinality=%s"
                            for cardinality in relations:
                                domain.add_value(feature % cardinality)

        domain.freeze()
        return domain

    def get_obj_props(self) -> List[Tuple[bytes, bytes]]:
        obj_props: Set[Tuple[bytes, bytes]] = set()
        for sm in self.training_models:
            for link in sm.graph.iter_links():
                obj_props.add((link.get_source_node().label, link.label))
        return list(obj_props)


def numbering_link_labels(links: List[GraphLink]) -> Dict[int, int]:
    accum_numbered_links = {}
    numbered_links = {}

    for l in links:
        if l.label not in accum_numbered_links:
            accum_numbered_links[l.label] = 1
        else:
            accum_numbered_links[l.label] += 1

    for l in links:
        numbered_links[l.id] = accum_numbered_links[l.label]
        accum_numbered_links[l.label] -= 1

    return numbered_links
