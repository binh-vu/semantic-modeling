#!/usr/bin/python
# -*- coding: utf-8 -*-
from numpy import percentile
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind

from semantic_labeling.column import Column


def ks_test(col1: Column, col2: Column) -> float:
    if len(col1.get_numeric_data()) > 1 and len(col2.get_numeric_data()) > 1:
        return ks_2samp(col1.get_numeric_data(), col2.get_numeric_data())[1]
    return 0.0


def mann_whitney_u_test(col1: Column, col2: Column) -> float:
    if len(col1.get_numeric_data()) > 1 and len(col2.get_numeric_data()) > 1:
        return mannwhitneyu(col1.get_numeric_data(), col2.get_numeric_data())[1]
    return 0.0


def welch_test(col1: Column, col2: Column) -> float:
    if len(col1.get_numeric_data()) > 1 and len(col2.get_numeric_data()) > 1:
        return ttest_ind(col1.get_numeric_data(), col2.get_numeric_data())[1]
    return 0.0


def jaccard_sim_test(col1: Column, col2: Column) -> float:
    col1data = set(col1.get_numeric_data())
    col2data = set(col2.get_numeric_data())

    if len(col2data) == 0 or len(col1data) == 0:
        return 0

    return len(col1data.intersection(col2data)) / len(col1data.union(col2data))


def coverage_test(col1: Column, col2: Column) -> float:
    col1_numeric_data = col1.get_numeric_data()
    col2_numeric_data = col2.get_numeric_data()

    if len(col1_numeric_data) > 1 and len(col2_numeric_data) > 1:
        max1 = percentile(col1_numeric_data, 100)
        min1 = percentile(col1_numeric_data, 0)
        max2 = percentile(col2_numeric_data, 100)
        min2 = percentile(col2_numeric_data, 0)
        max3 = max(max1, max2)
        min3 = min(min1, min2)

        if min2 > max1 or min1 > max2 or max3 == min3:
            return 0.0
        else:
            max4 = min(max1, max2)
            min4 = max(min1, min2)
            return (max4 - min4) / (max3 - min3)

    return 0.0




