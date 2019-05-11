#!/usr/bin/python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple, List, Optional

from dateutil.parser import parse as parse_date
from pathlib import Path

from data_structure import Graph, GraphLink, GraphNode
from semantic_labeling.column import Column, ColumnType
from semantic_labeling.column_based_table import ColumnBasedTable
from semantic_modeling.data_io import get_cache_dir, get_sampled_data_tables
from semantic_modeling.settings import Settings
from semantic_modeling.config import get_logger, config
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import deserialize, serialize


class DataConstraint(object):
    """This model tries to answer a question whether a mapping of a column follow some constraints inferred from data

    For example: we have 2 columns DoB & DoD
        1. if they are linked to same class by different predicate (local constraint)
        2. if they are linked to same class by same predicate, which class should we choose to link? (look at the parent)
        + how about: the case of same predicate, different class (not handled yet, let's semantic labeling does it)

    Given a list of known sources, we extract possible columns order =, >=, <= (consider only columns that are in the above scope).
    The we count cases the semantic types/relationship follow the order, and cases it doesn't as prob.

    Given a new example, we also look for columns that match, and produce the prediction
    """

    logger = get_logger("app.weak_models.data_constraint")

    def __init__(self, train_sms: List[SemanticModel], data_tables: List[ColumnBasedTable], valid_threshold: float,
                 guess_datetime_threshold: float, n_comparison_sample: int) -> None:
        self.guess_datetime_threshold = guess_datetime_threshold
        self.valid_threshold = valid_threshold
        self.n_comparison_sample = n_comparison_sample
        self.cached_compared_cols: Dict[str, Dict[Tuple[bytes, bytes], Optional[float]]] = {}
        self.prob_count_scope1: Dict[Tuple[bytes, bytes], Dict[Tuple[bytes, bytes], int]] = {}
        self.prob_count_scope2: Dict[Tuple[bytes, bytes], Dict[Tuple[bytes, bytes], int]] = {}

        # keep a list of columns that can have data constraint (i.e: its value is comparable with other columns)
        col2useful_type: Dict[Column, ColumnType] = {}
        data_tables: Dict[str, ColumnBasedTable] = {tbl.id: tbl for tbl in data_tables}
        for tbl in data_tables.values():
            for col in tbl.columns:
                type = self._guess_detail_type(col)
                if type is not None and type.is_comparable():
                    col2useful_type[col] = type

        # now we build the constraint from training sources
        for sm in train_sms:
            stypes: Dict[Tuple[bytes, bytes], List[GraphLink]] = {}
            node_group: Dict[GraphNode, List[GraphLink]] = {}
            table = data_tables[sm.id]
            name2col: Dict[bytes, Column] = {col.name.encode("utf-8"): col for col in table.columns}
            col2idx: Dict[Column, int] = {col: i for i, col in enumerate(table.columns)}

            for attr in sm.attrs:
                dnode = sm.graph.get_node_by_id(attr.id)
                dlink = dnode.get_first_incoming_link()
                pnode = dlink.get_source_node()
                stype = (pnode.label, dlink.label)
                if stype not in stypes:
                    stypes[stype] = []
                stypes[stype].append(dlink)
                # group node by their parents
                if pnode not in node_group:
                    node_group[pnode] = []
                node_group[pnode].append(dlink)

            # first scope, infer constraint inside class nodes
            for pnode, dlinks in node_group.items():
                # before filter out data nodes that are not comparable, we double check if the data node we
                # have to ignore has its semantic types comparable
                # for e in dlinks:
                #     if name2col[e.get_target_node().label] not in col2useful_type and (pnode.label, e.label) in self.prob_count_scope1:
                #         self.logger.warning("Column's semantic types was detected to be comparable. But, now it can't: %s: %s", sm.id, e.get_target_node().label)
                #         self.prob_count_scope1[(pnode.label, e.label)] = None

                dlinks = [e for e in dlinks if name2col[e.get_target_node().label] in col2useful_type]
                dnodes = [e.get_target_node() for e in dlinks]
                if len(dnodes) < 2:
                    continue

                if len({col2useful_type[name2col[dnode.label]] for dnode in dnodes}) != 1:
                    # doesn't support mixed-type
                    print({col2useful_type[name2col[dnode.label]] for dnode in dnodes})
                    continue

                if len(dnodes) > 2:
                    self.logger.warning("Only handle max-2 now... %s: %s", sm.id, [e.label for e in dlinks])
                    continue

                cols = [name2col[dnode.label] for dnode in dnodes]
                compare_result = self._compare_col(table, col2idx, cols[0], cols[1])

                dtypes = [(pnode.label, dlink.label) for dlink in dlinks]
                # if we cannot compare 2 columns, then we ignore them
                if compare_result is None:
                    # however, if this type is already register in the counter, then instead of ignore them
                    # we should delete set it to None to prevent re-add in the future
                    if (dtypes[0] in self.prob_count_scope1 and self.prob_count_scope1[dtypes[0]] is not None) or (
                            dtypes[1] in self.prob_count_scope1 and self.prob_count_scope1[dtypes[1]] is not None):
                        self.logger.warning(
                            "Inferred constraint for 2 columns %s doesn't hold for source: %s. (Column: %s, %s)",
                            dtypes, sm.id, cols[0].name, cols[1].name)
                        self.prob_count_scope1[dtypes[0]] = None
                        self.prob_count_scope1[dtypes[1]] = None
                    continue

                for dtype in dtypes:
                    if dtype not in self.prob_count_scope1:
                        self.prob_count_scope1[dtype] = {}

                if self.prob_count_scope1[dtypes[0]] is None or self.prob_count_scope1[dtypes[1]] is None:
                    # inferred constraint doesn't hold, so we should ignore this column
                    continue

                if compare_result:
                    # col0 > col1
                    if len(self.prob_count_scope1[dtypes[0]]) != 0:
                        if dtypes[0] not in self.prob_count_scope1[dtypes[0]] or dtypes[1] not in self.prob_count_scope1[dtypes[0]]:
                            self.prob_count_scope1[dtypes[0]] = None
                            self.prob_count_scope1[dtypes[1]] = None
                        else:
                            assert self.prob_count_scope1[dtypes[0]][dtypes[0]] == 1 and self.prob_count_scope1[dtypes[0]][dtypes[1]] == 0
                    if len(self.prob_count_scope1[dtypes[1]]) != 0:
                        if dtypes[0] not in self.prob_count_scope1[dtypes[1]] or dtypes[1] not in self.prob_count_scope1[dtypes[1]]:
                            self.prob_count_scope1[dtypes[0]] = None
                            self.prob_count_scope1[dtypes[1]] = None
                        else:
                            assert self.prob_count_scope1[dtypes[1]][dtypes[0]] == 1 and self.prob_count_scope1[dtypes[1]][dtypes[1]] == 0
                    self.prob_count_scope1[dtypes[0]] = {dtypes[0]: 1, dtypes[1]: 0}
                    self.prob_count_scope1[dtypes[1]] = {dtypes[0]: 1, dtypes[1]: 0}
                else:
                    if len(self.prob_count_scope1[dtypes[0]]) != 0:
                        assert self.prob_count_scope1[dtypes[0]][dtypes[0]] == 0 and self.prob_count_scope1[dtypes[0]][dtypes[1]] == 1
                    if len(self.prob_count_scope1[dtypes[1]]) != 0:
                        assert self.prob_count_scope1[dtypes[1]][dtypes[0]] == 0 and self.prob_count_scope1[dtypes[1]][dtypes[1]] == 1
                    self.prob_count_scope1[dtypes[0]] = {dtypes[0]: 0, dtypes[1]: 1}
                    self.prob_count_scope1[dtypes[1]] = {dtypes[0]: 0, dtypes[1]: 1}

            # second scope
            for stype, dlinks in stypes.items():
                if len(dlinks) == 1:
                    continue

                # now filter data nodes that is not comparable
                dnodes = [e.get_target_node() for e in dlinks]
                if any(name2col[dnode.label] not in col2useful_type for dnode in dnodes):
                    continue

                if len({col2useful_type[name2col[dnode.label]] for dnode in dnodes}) != 1:
                    # doesn't support mixed-type
                    print({col2useful_type[name2col[dnode.label]] for dnode in dnodes})
                    continue

                if len(dlinks) > 2:
                    self.logger.warning("Only handle max-2 now... %s: %s", sm.id, stype)
                    continue

                snodes = [e.get_source_node() for e in dlinks]
                slinks = [n.get_first_incoming_link() for n in snodes if n.get_first_incoming_link() is not None]
                if len(slinks) == 0:
                    continue

                # now we need to build some constraints to help distinguish between those semantic types
                # we assume parents of those types are different ...
                parent_types = [(se.get_source_node().label, se.label) for se in slinks]
                if len(set(parent_types)) != len(snodes):
                    self.logger.warning("Doesn't handle a case when parents are same: %s: %s", sm.id, stype)
                    continue

                cols = [name2col[dnode.label] for dnode in dnodes]
                compare_result = self._compare_col(table, col2idx, cols[0], cols[1])
                # if we cannot compare 2 columns, then we ignore them
                if compare_result is None:
                    # however, if this type is already register in the counter, then instead of ignore them
                    # we should delete set it to None to prevent re-add in the future
                    if stype in self.prob_count_scope2 and self.prob_count_scope2[stype] is not None:
                        self.logger.warning(
                            "Inferred constraint for type %s doesn't hold for source: %s. (Column: %s, %s)", stype,
                            sm.id, cols[0].name, cols[1].name)
                        self.prob_count_scope2[stype] = None
                    continue

                if stype not in self.prob_count_scope2:
                    self.prob_count_scope2[stype] = {}

                if self.prob_count_scope2[stype] is None:
                    # inferred constraint doesn't hold, so we should ignore this column
                    continue

                if compare_result:
                    # col0 > col1
                    if len(self.prob_count_scope2[stype]) != 0:
                        assert self.prob_count_scope2[stype][parent_types[0]] == 1 and self.prob_count_scope2[stype][parent_types[1]] == 0
                    self.prob_count_scope2[stype] = {parent_types[0]: 1, parent_types[1]: 0}
                else:
                    if len(self.prob_count_scope2[stype]) != 0:
                        assert self.prob_count_scope2[stype][parent_types[0]] == 0 and self.prob_count_scope2[stype][parent_types[1]] == 1
                    self.prob_count_scope2[stype] = {parent_types[0]: 0, parent_types[1]: 1}

        for key in list(self.prob_count_scope1.keys()):
            if self.prob_count_scope1[key] is None:
                del self.prob_count_scope1[key]

        for key in list(self.prob_count_scope2.keys()):
            if self.prob_count_scope2[key] is None:
                del self.prob_count_scope2[key]

        # we also cache column comparison (to speed to evaluation time)
        for tbl in data_tables.values():
            useful_cols = [col for col in tbl.columns if col in col2useful_type]
            tbl_comparison: Dict[Tuple[bytes, bytes], Optional[float]] = {}
            col2idx: Dict[Column, int] = {col: i for i, col in enumerate(tbl.columns)}
            # TODO: can speed up by half
            for col in useful_cols:
                col_name = col.name.encode("utf-8")
                for col2 in useful_cols:
                    if col2 != col:
                        if col2useful_type[col] != col2useful_type[col2]:
                            tbl_comparison[(col_name, col2.name.encode("utf-8"))] = None
                        else:
                            tbl_comparison[(col_name, col2.name.encode("utf-8"))] = self._compare_col(
                                tbl, col2idx, col, col2)

            self.cached_compared_cols[tbl.name] = tbl_comparison

    def extract_feature(self, sm_id: str, g: Graph, attr_id: int, link2label: Optional[Dict[int, bool]] = None) -> dict:
        return {
            "local": self.compute_prob_scope1(sm_id, g, attr_id, link2label),
            "global": self.compute_prob_scope2(sm_id, g, attr_id, link2label),
        }

    def compute_prob_scope1(self, sm_id: str, g: Graph, attr_id: int,
                            link2label: Optional[Dict[int, bool]] = None) -> Optional[float]:
        if link2label is None:
            # use default dict to reduce code size
            link2label = {}
        dnode = g.get_node_by_id(attr_id)
        dlink = dnode.get_first_incoming_link()
        pnode = dlink.get_source_node()
        stype = (pnode.label, dlink.label)
        if stype not in self.prob_count_scope1 or not link2label.get(dlink.id, True):
            return None

        assert len(self.prob_count_scope1[stype]) == 2
        another_stype = [x for x in self.prob_count_scope1[stype].keys() if x != stype][0]
        another_dnodes = [
            e.get_target_node() for e in pnode.iter_outgoing_links()
            if e.label == another_stype[1] and link2label.get(e.id, True)
        ]
        if len(another_dnodes) == 0:
            return None

        dnode_stype_idx = self.prob_count_scope1[stype][stype]
        another_dnode_stype_idx = self.prob_count_scope1[stype][another_stype]
        tbl_comparison = self.cached_compared_cols[sm_id]
        result = None

        for another_dnode in another_dnodes:
            if (dnode.label, another_dnode.label) not in tbl_comparison:
                continue

            result = tbl_comparison[(dnode.label, another_dnode.label)]
            if result is None:
                continue

            if result:
                # attr > another_attr, attr_stype_idx should > another_attr_stype_idx with high prob.
                if dnode_stype_idx > another_dnode_stype_idx:
                    return self.valid_threshold
                return 1 - self.valid_threshold
            else:
                # opposite case of above
                if dnode_stype_idx > another_dnode_stype_idx:
                    return 1 - self.valid_threshold
                return self.valid_threshold

        if result is None:
            # the constraint said that we should be able to compare, but we cannot, it should have low probability
            return 1 - self.valid_threshold

    def compute_prob_scope2(self, sm_id: str, g: Graph, attr_id: int,
                            link2label: Optional[Dict[int, bool]] = None) -> Optional[float]:
        """Give a probability whether mapping of an attribute statistic data constraints

        We can mark some part of graph as false
        """
        dnode = g.get_node_by_id(attr_id)
        dlink = dnode.get_first_incoming_link()
        stype = (dlink.get_source_node().label, dlink.label)
        if stype not in self.prob_count_scope2:
            return None

        slink = dlink.get_source_node().get_first_incoming_link()
        if slink is None:
            # root nodes
            return None

        dnode_parent_type = (slink.get_source_node().label, slink.label)
        if dnode_parent_type not in self.prob_count_scope2[stype] or (link2label is not None
                                                                      and not link2label[slink.id]):
            return None
        dnode_stype_idx = self.prob_count_scope2[stype][dnode_parent_type]

        # get other class nodes in the graph that an attr can be mapped to (same semantic type).
        # notice that the constraint is represent as binary-function, so we only keep the class nodes
        # that have another attribute, which is mapped with the same semantic type
        snodes = [node for node in g.iter_nodes_by_label(stype[0]) if node.id != dlink.source_id]
        if len(snodes) == 0:
            # if we don't have any other source nodes (i.e: only one possible mapping)
            return None

        tbl_comparison = self.cached_compared_cols[sm_id]
        another_dnodes = []
        another_dnodes_stype_idx = []
        for snode in snodes:
            # check if this source node have another attribute that is mapped by same semantic type
            for link in snode.iter_outgoing_links():
                if link.label == dlink.label:
                    another_dnode = link.get_target_node()
                    break
            else:
                another_dnode = None

            if another_dnode is not None and (dnode.label, another_dnode.label) in tbl_comparison:
                slink = snode.get_first_incoming_link()
                parent_type = (slink.get_source_node().label, slink.label)
                if parent_type in self.prob_count_scope2[stype] and (link2label is None
                                                                     or link2label[slink.id] is True):
                    # if its parent_type is not in the constraint or its link is false, then we should ignore it
                    another_dnodes.append(another_dnode)
                    another_dnodes_stype_idx.append(self.prob_count_scope2[stype][parent_type])

        # do compare between attr and another_attrs
        if len(another_dnodes) + 1 > len(self.prob_count_scope2[stype]):
            self.logger.warning(
                "There is a model that have more attributes than the inferred constraint.. trace: %s -- %s", sm_id,
                stype)
            return None

        # let's see if we can compare the given attribute with other attributes
        if len(another_dnodes) == 0 or dnode_stype_idx in another_dnodes_stype_idx:
            # how about this case?
            return None

        assert len(self.prob_count_scope2[stype]) == 2, "Doesn't handle > 2 attributes now..."

        # now we can compare with other attributes
        another_dnode, another_dnode_stype_idx = another_dnodes[0], another_dnodes_stype_idx[0]
        result = tbl_comparison[(dnode.label, another_dnode.label)]
        if result is None:
            # the constraint said that we should be able to compare, but we cannot, it should have low probability
            return 1 - self.valid_threshold

        if result:
            # attr > another_attr, attr_stype_idx should > another_attr_stype_idx with high prob.
            if dnode_stype_idx > another_dnode_stype_idx:
                return self.valid_threshold
            return 1 - self.valid_threshold
        else:
            # opposite case of above
            if dnode_stype_idx > another_dnode_stype_idx:
                return 1 - self.valid_threshold
            return self.valid_threshold

    def _compare_col(self, tbl: ColumnBasedTable, col2idx, col1: Column, col2: Column) -> Optional[bool]:
        # any mixed-type should be handled before..
        n_gt, n_eq, n_lt = 0, 0, 0
        count = 0
        if col1.type == ColumnType.NUMBER:
            for row in tbl.rows:
                val1 = row[col2idx[col1]]
                val2 = row[col2idx[col2]]

                if not isinstance(val1, (int, float)) or not isinstance(val2,
                                                                        (int, float)) or val1 is None or val2 is None:
                    continue

                if val1 == val2:
                    n_eq += 1
                elif val1 > val2:
                    n_gt += 1
                else:
                    n_lt += 1

                count += 1
                if count == self.n_comparison_sample:
                    break
        else:
            for row in tbl.rows:
                val1 = row[col2idx[col1]]
                val2 = row[col2idx[col2]]
                if not isinstance(val1, (str, bytes)) or not isinstance(val2,
                                                                        (str, bytes)) or val1 is None or val2 is None:
                    continue

                try:
                    # TODO: need to detect it is
                    val1 = parse_date(val1, dayfirst=False, yearfirst=False)
                    val2 = parse_date(val2, dayfirst=False, yearfirst=False)
                except ValueError:
                    continue

                if val1 == val2:
                    n_eq += 1
                elif val1 > val2:
                    n_gt += 1
                else:
                    n_lt += 1

                count += 1
                if count == 50:
                    break

        if n_gt > 0 and ((n_gt + n_eq) / count) >= self.valid_threshold:
            return True
        if n_lt > 0 and ((n_lt + n_eq) / count) >= self.valid_threshold:
            return False

        # not decidable (also for equal-case)
        return None

    def _guess_detail_type(self, col: Column):
        if col.type == ColumnType.NUMBER:
            return ColumnType.NUMBER
        if col.type == ColumnType.NULL:
            return None

        # trying to guess if this is DateTime
        # just get first 100 values to reduce computing time
        values = [val for val in col.get_textual_data() if val.strip() != ""][:50]
        n_success = 0
        for val in values:
            try:
                parse_date(val)
                n_success += 1
            except ValueError:
                pass

        if (n_success / len(values)) > self.guess_datetime_threshold:
            # consider this is a datetime column
            return ColumnType.DATETIME
        return None


_instance = None


def get_data_constraint_model(
        dataset: str,
        train_sms: List[SemanticModel],
) -> DataConstraint:
    global _instance
    if _instance is None:
        cache_file = get_cache_dir(dataset, train_sms) / "weak_models" / "data_constraint.pkl"
        cache_file.parent.mkdir(exist_ok=True, parents=True)
        need_rebuilt = True

        settings = Settings.get_instance()
        valid_threshold = settings.data_constraint_valid_threshold
        guess_datetime_threshold = settings.data_constraint_guess_datetime_threshold
        n_comparison_samples = settings.data_constraint_n_comparison_samples
        random_seed = settings.random_seed
        n_sample = settings.n_samples

        if cache_file.exists():
            DataConstraint.logger.debug("Try to load previous run...")
            model, cached_dataset, cached_train_sm_ids, extra_args = deserialize(cache_file)
            if cached_dataset == dataset \
                    and cached_train_sm_ids == {sm.id for sm in train_sms} \
                    and extra_args == (
                        valid_threshold, guess_datetime_threshold, n_comparison_samples,
                        random_seed, n_sample):
                need_rebuilt = False

        if need_rebuilt:
            DataConstraint.logger.debug("Re-build data-constraint model...")
            data_tables = [ColumnBasedTable.from_table(tbl) for tbl in get_sampled_data_tables(dataset)]
            model = DataConstraint(train_sms, data_tables, valid_threshold, guess_datetime_threshold,
                                   n_comparison_samples)
            serialize((model, dataset, {sm.id
                                        for sm in train_sms},
                       (valid_threshold, guess_datetime_threshold, n_comparison_samples, random_seed, n_sample)),
                      cache_file)

        _instance = model
    return _instance
