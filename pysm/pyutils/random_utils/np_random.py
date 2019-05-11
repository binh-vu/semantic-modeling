#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import operator


class RandomState(object):

    def __init__(self, seed=None) -> None:
        self.seed = seed
        self.random_state = np.random.RandomState(seed=seed)

        self.choice_a = None
        self.choice_p = None
        self.choice_p_backup = None

        self.choice_a_size = None
        self.choice_p_nonzero_count = None
        self.choice_cdf = None

    def set_choice_parameters(self, a, p) -> 'RandomState':
        """
            Set parameter for self.choice function (which is optimization version of numpy.random.choice)
            :param a:
            :param p:
            :return:
        """
        # DISCLAIMER: this code copied from https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx

        # Format and Verify input
        a = np.array(a, copy=False)
        if a.ndim == 0:
            try:
                # __index__ must return an integer by python rules.
                pop_size = operator.index(a.item())
            except TypeError:
                raise ValueError("a must be 1-dimensional or an integer")
            if pop_size <= 0:
                raise ValueError("a must be greater than 0")
        elif a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        else:
            pop_size = a.shape[0]
            if pop_size is 0:
                raise ValueError("a must be non-empty")

        if p is not None:
            atol = np.sqrt(np.finfo(np.float64).eps)
            if isinstance(p, np.ndarray):
                if np.issubdtype(p.dtype, np.float):
                    atol = max(atol, np.sqrt(np.finfo(p.dtype).eps))

            p = np.asarray(p, dtype=np.double)

            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("a and p must have same size")
            # noinspection PyArgumentList
            if np.logical_or.reduce(p < 0):
                raise ValueError("probabilities are not non-negative")
            if abs(np.sum(p) - 1.) > atol:
                raise ValueError("probabilities do not sum to 1")

            cdf = p.cumsum()
            cdf /= cdf[-1]

            self.choice_p_backup = p.copy()
            self.choice_p_nonzero_count = np.count_nonzero(p > 0)
            self.choice_cdf = cdf

        # MY CODE
        self.choice_a_size = pop_size
        self.choice_a = a
        self.choice_p = p
        return self

    def uniform_choice(self, array, size, replace=True):
        if replace is True:
            raise NotImplemented('not implemented yet')
        else:
            assert size < len(array)
            result = set()
            while len(result) < size:
                result.add(self.random_state.randint(0, len(array)))

            return map(lambda x: array[x], result)

    def choice(self, size=None, replace=True):
        """
            Opt. version of numpy.random.choice with parameter a & p is set by self.set_choice_parameters func

            :param size:
            :param replace:
            :return:
        """
        shape = size
        if shape is not None:
            size = np.prod(shape, dtype=np.intp)
        else:
            size = 1

        # Actual sampling
        if replace:
            if self.choice_p is not None:
                uniform_samples = self.random_state.random_sample(shape)
                idx = self.choice_cdf.searchsorted(uniform_samples, side='right')
                idx = np.array(idx, copy=False)  # searchsorted returns a scalar
            else:
                idx = self.random_state.randint(0, self.choice_a_size, size=shape)
        else:
            if size > self.choice_a_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when 'replace=False'")

            if self.choice_p is not None:
                if self.choice_p_nonzero_count < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                found = np.zeros(shape, dtype=np.int)
                flat_found = found.ravel()
                while n_uniq < size:
                    # noinspection PyArgumentList
                    x = self.random_state.rand(size - n_uniq)
                    if n_uniq > 0:
                        self.choice_p[flat_found[0:n_uniq]] = 0
                        cdf = np.cumsum(self.choice_p)
                        cdf /= cdf[-1]
                    else:
                        cdf = self.choice_cdf

                    new = cdf.searchsorted(x, side='right')
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq:n_uniq + new.size] = new
                    n_uniq += new.size
                # reset the choice p so we don't need to perform a copy operation
                self.choice_p[flat_found] = self.choice_p_backup[flat_found]
                idx = found
            else:
                raise Exception("not implemented yet")
                # idx = self.random_state.permutation(self.choice_a_size)[:size]
                # if shape is not None:
                #     idx.shape = shape

        if shape is None and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        # Use samples as indices for a if a is array-like
        if self.choice_a.ndim == 0:
            return idx

        if shape is not None and idx.ndim == 0:
            # If size == () then the user requested a 0-d array as opposed to
            # a scalar object when size is None. However a[idx] is always a
            # scalar and not an array. So this makes sure the result is an
            # array, taking into account that np.array(item) may not work
            # for object arrays.
            res = np.empty((), dtype=self.choice_a.dtype)
            res[()] = self.choice_a[idx]
            return res

        return self.choice_a[idx]
