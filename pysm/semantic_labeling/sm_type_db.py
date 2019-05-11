#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict, Counter
from itertools import chain
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Dict, Tuple, List

import numpy

from data_structure import GraphNode
from semantic_labeling.column import Column
from semantic_labeling.column_based_table import ColumnBasedTable
from semantic_labeling.feature_extraction.column_base import numeric, textual, column_name
from semantic_modeling.config import get_logger
from semantic_modeling.data_io import get_semantic_models, get_sampled_data_tables
from semantic_modeling.karma.semantic_model import SemanticModel
from semantic_modeling.utilities.serializable import deserialize, serialize


class SemanticTypeDB(object):

    logger = get_logger('app.semantic_labeling.stype_db')
    SIMILARITY_METRICS = [
        "label_jaccard", "stype_jaccard", "num_ks_test", "num_mann_whitney_u_test", "num_jaccard", "text_jaccard", "text_tf-idf"
    ]
    instance = None

    def __init__(self, dataset: str, train_tables: List[ColumnBasedTable], test_tables: List[ColumnBasedTable]):
        self.dataset = dataset
        self.train_tables = train_tables
        self.test_tables = test_tables

        self.similarity_matrix: numpy.ndarray = None
        self.tfidf_db: TfidfDatabase = None
        self._init()

    def _init(self):
        self.source_mappings: Dict[str, SemanticModel] = {s.id: s for s in get_semantic_models(self.dataset)}
        self.train_columns = [col for tbl in self.train_tables for col in tbl.columns]
        self.train_column_stypes: List[str] = []
        for tbl in self.train_tables:
            sm = self.source_mappings[tbl.id]
            for col in tbl.columns:
                dnode = sm.graph.get_node_by_id(sm.get_attr_by_label(col.name).id)
                dlink = dnode.get_first_incoming_link()
                self.train_column_stypes.append(dlink.label.decode("utf-8"))

        self.test_columns = [col for tbl in self.test_tables for col in tbl.columns]
        self.name2table: Dict[str, ColumnBasedTable] = {
            tbl.id: tbl for tbl in chain(self.train_tables, self.test_tables)
        }
        self.col2idx: Dict[str, int] = {col.id: i for i, col in enumerate(chain(self.train_columns, self.test_columns))}
        self.col2types: Dict[str, Tuple[str, str]] = {}
        self.col2dnodes: Dict[str, GraphNode] = {}

        col: Column
        for col in chain(self.train_columns, self.test_columns):
            sm = self.source_mappings[col.table_name]
            attr = sm.get_attr_by_label(col.name)
            dnode = sm.graph.get_node_by_id(attr.id)
            link = dnode.get_first_incoming_link()
            self.col2types[col.id] = (link.get_source_node().label, link.label)
            self.col2dnodes[col.id] = dnode

        assert len(self.col2types) == len(self.train_columns) + len(self.test_columns), "column name must be unique"

    @staticmethod
    def create(dataset: str, train_source_ids: List[str]) -> 'SemanticTypeDB':
        tables = get_sampled_data_tables(dataset)
        train_source_ids = set(train_source_ids)

        train_tables = [ColumnBasedTable.from_table(tbl) for tbl in tables if tbl.id in train_source_ids]
        test_tables = [ColumnBasedTable.from_table(tbl) for tbl in tables if tbl.id not in train_source_ids]

        return SemanticTypeDB(dataset, train_tables, test_tables)

    @staticmethod
    def get_stype_db(dataset: str, train_source_ids: List[str], cache_dir: Path) -> 'SemanticTypeDB':
        if SemanticTypeDB.instance is None:
            cache_file = cache_dir / 'stype_db.pkl'
            if cache_file.exists():
                SemanticTypeDB.logger.debug("Load SemanticTypeDB from cache file...")
                stype_db: SemanticTypeDB = deserialize(cache_file)
                if set(train_source_ids) != {tbl.id for tbl in stype_db.train_tables} or stype_db.dataset != dataset:
                    stype_db = None
            else:
                stype_db = None

            if stype_db is None:
                SemanticTypeDB.logger.debug("Have to re-create SemanticTypeDB...")
                stype_db = SemanticTypeDB.create(dataset, train_source_ids)
                stype_db._build_db()
                serialize(stype_db, cache_file)

            SemanticTypeDB.instance = stype_db

        return SemanticTypeDB.instance

    def get_table_by_name(self, name: str) -> ColumnBasedTable:
        return self.name2table[name]

    def _build_db(self) -> None:
        """Build semantic types database from scratch"""
        n_train_columns = len(self.train_columns)

        self.logger.debug("Build tfidf database...")
        self.similarity_matrix = numpy.zeros(
            (n_train_columns + len(self.test_columns), n_train_columns, len(self.SIMILARITY_METRICS)), dtype=float)
        self.tfidf_db = TfidfDatabase.create(textual.get_tokenizer(), self.train_columns)

        self.logger.debug("Pre-build tf-idf for all columns")
        self.tfidf_db.cache_tfidf(self.test_columns)
        self.logger.debug("Computing similarity matrix...")

        # loop through train source ids and compute similarity between columns
        for idx, col in enumerate(self.train_columns):
            self.logger.trace("   + working on col: %s", col.id)
            sim_features = self._compute_feature_vectors(col, self.train_columns, self.train_column_stypes)
            self.similarity_matrix[idx, :, :] = numpy.asarray(sim_features).reshape((n_train_columns, -1))

        for idx, col in enumerate(self.test_columns):
            self.logger.trace("   + working on col: %s", col.id)
            sim_features = self._compute_feature_vectors(col, self.train_columns, self.train_column_stypes)
            self.similarity_matrix[idx + n_train_columns, :, :] = numpy.asarray(sim_features).reshape((n_train_columns,
                                                                                                       -1))

    def _compute_feature_vectors(self, col: Column, refcols: List[Column], refcol_stypes: List[str]):
        features = []
        for i, refcol in enumerate(refcols):
            features.append([
                # name features
                column_name.jaccard_sim_test(refcol.name, col.name, lower=True),
                column_name.jaccard_sim_test(refcol_stypes[i], col.name, lower=True),
                # numeric features
                numeric.ks_test(refcol, col),
                numeric.mann_whitney_u_test(refcol, col),
                numeric.jaccard_sim_test(refcol, col),
                # text features
                textual.jaccard_sim_test(refcol, col),
                textual.cosine_similarity(self.tfidf_db.compute_tfidf(refcol), self.tfidf_db.compute_tfidf(col)),
            ])

        return features

    # implement pickling
    def __getstate__(self):
        return self.dataset, self.train_tables, self.test_tables, self.similarity_matrix

    def __setstate__(self, state):
        self.dataset = state[0]
        self.train_tables = state[1]
        self.test_tables = state[2]
        self.similarity_matrix = state[3]
        self._init()


class TfidfDatabase(object):

    logger = get_logger('app.semantic_labeling.tfidf_db')

    def __init__(self, tokenizer, vocab: Dict[str, int], invert_token_idx: Dict[str, int],
                 col2tfidf: Dict[str, numpy.ndarray]) -> None:
        self.vocab = vocab
        self.invert_token_idx = invert_token_idx
        self.tokenizer = tokenizer
        self.n_docs = len(col2tfidf)
        self.cache_col2tfidf = col2tfidf

    @staticmethod
    def create(tokenizer, columns: List[Column]) -> 'TfidfDatabase':
        vocab = {}
        invert_token_idx: Dict[str, int] = defaultdict(lambda: 0)
        col2tfidf = {}
        token_count = defaultdict(lambda: 0)
        n_docs = len(columns)

        # compute tf first
        with Pool() as p:
            tf_cols = p.map(TfidfDatabase._compute_tf, [(tokenizer, col) for col in columns])

        # then compute vocabulary & preparing for idf
        for tf_col in tf_cols:
            for w in tf_col:
                invert_token_idx[w] += 1
                token_count[w] += 1

        # reduce vocab size
        for w in token_count:
            if token_count[w] < 2 and w.isdigit():
                # delete this word
                del invert_token_idx[w]
            else:
                vocab[w] = len(vocab)

        # revisit it and make tfidf
        for col, tf_col in zip(columns, tf_cols):
            tfidf = numpy.zeros((len(vocab)))
            for w, tf in tf_col.items():
                if w in vocab:
                    tfidf[vocab[w]] = tf * numpy.log(n_docs / (1 + invert_token_idx[w]))
            col2tfidf[col.id] = tfidf

        return TfidfDatabase(tokenizer, vocab, invert_token_idx, col2tfidf)

    def compute_tfidf(self, col: Column):
        if col.id in self.cache_col2tfidf:
            return self.cache_col2tfidf[col.id]

        tfidf = numpy.zeros(len(self.vocab))
        for w, tf in self._compute_tf((self.tokenizer, col)).items():
            if w in self.vocab:
                print(w, tf, self.invert_token_idx[w], numpy.log(self.n_docs / (1 + self.invert_token_idx[w])))
                tfidf[self.vocab[w]] = tf * numpy.log(self.n_docs / (1 + self.invert_token_idx[w]))

        return tfidf

    def  cache_tfidf(self, cols: List[Column]):
        cols = [col for col in cols if col.id not in self.cache_col2tfidf]

        with Pool() as p:
            tf_cols = p.map(TfidfDatabase._compute_tf, [(self.tokenizer, col) for col in cols])

        for col, tf_col in zip(cols, tf_cols):
            tfidf = numpy.zeros(len(self.vocab))
            for w, tf in tf_col.items():
                if w in self.vocab:
                    tfidf[self.vocab[w]] = tf * numpy.log(self.n_docs / (1 + self.invert_token_idx[w]))
            self.cache_col2tfidf[col.id] = tfidf

    @staticmethod
    def _compute_tf(args):
        tokenizer, col = args
        counter = Counter()
        sents = (subsent for sent in col.get_textual_data() for subsent in sent.decode('utf-8').split("/"))
        for doc in tokenizer.pipe(sents, batch_size=50, n_threads=4):
            counter.update((str(w) for w in doc))

        number_of_token = sum(counter.values())
        for token, val in counter.items():
            counter[token] = val / number_of_token
        return counter


if __name__ == '__main__':
    stype_db = SemanticTypeDB.create("museum_edm", [sm.id for sm in get_semantic_models("museum_edm")[:14]])
    stype_db._build_db()