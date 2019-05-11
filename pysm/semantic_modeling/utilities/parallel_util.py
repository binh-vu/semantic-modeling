#!/usr/bin/python
# -*- coding: utf-8 -*-
from multiprocessing.pool import Pool
from typing import Dict, Tuple, List, Set, Union, Optional, Callable, Generic, TypeVar

from multiprocessing import Process, Queue, get_start_method, set_start_method

import time

import os
from nose.tools import eq_
from pyutils.progress_utils import Timer

from semantic_modeling.config import get_logger
"""Provide an easy and quick way to test if parallel is worth to do (overhead cost of serialize/deserialize arguments)"""
logger = get_logger("default")


def get_args_size(*args) -> int:
    total_element = 0
    for arg in args:
        if isinstance(arg, (list, dict, tuple)):
            total_element += len(arg)
        else:
            total_element += 1
    return total_element


def minimal_computing_func(queue, *args):
    """A function that doesn't do anything but use to test overhead cost of multiprocessing"""
    queue.put(get_args_size(*args))


def zero_computing_func(*args):
    return len(args)


def get_batchs(n_elements: int, n_batch: int) -> List[Tuple[int, int]]:
    batch_size = int(n_elements / n_batch)
    batchs = [(i * batch_size, (i + 1) * batch_size) for i in range(n_batch)]
    batchs[-1] = (batchs[-1][0], n_elements)
    return batchs


def benchmark_overhead_time(get_args: Callable[[], Tuple]):
    """Note: use a function to create arguments instead of passing through function arguments, because when we using fork,
    new processes also inherit arguments through copy not pickling.
    """
    queue = Queue()
    timer = Timer("ms").start()
    p = Process(target=minimal_computing_func, args=(queue, 'peter', "john"))
    p.start()
    eq_(queue.get(), 2)
    p.join()
    logger.info("Default overhead: %s", timer.lap().get_total_time())

    default_time = timer.total_time
    args = get_args()
    arg_size = get_args_size(*args)
    assert queue.empty()

    timer.reset()
    p = Process(target=minimal_computing_func, args=tuple([queue] + list(args)))
    p.start()
    eq_(queue.get(), arg_size)
    p.join()
    logger.info("Default overhead + serialize input takes: %s", timer.lap().get_total_time())
    logger.info("=> Overhead of sending input: %s ms", round((timer.total_time - default_time) * 1000, 4))


def benchmark_sending_time(get_args: Callable[[], Tuple]):
    def test(inqueue, outqueue):
        outqueue.put(len(inqueue.get()))

    inqueue = Queue()
    outqueue = Queue()
    args = get_args()
    n_args = len(args)

    timer = Timer("ms").start()
    inqueue.put(list(range(5)))
    p = Process(target=test, args=(inqueue, outqueue))
    p.start()
    res = outqueue.get()
    p.join()
    logger.info("Default time: %s", timer.lap().get_total_time())
    assert res == 5
    assert inqueue.empty() and outqueue.empty()

    default_time = timer.total_time
    timer.reset()
    inqueue.put(args)
    p = Process(target=test, args=(inqueue, outqueue))
    p.start()
    res = outqueue.get()
    p.join()
    logger.info("Default time + sending input: %s", timer.lap().get_total_time())
    assert res == n_args
    logger.info("=> Sending input: %s ms", round((timer.total_time - default_time) * 1000, 4))


def benchmark_overhead_pool_map_time(get_args: Callable[[], List[Tuple]], n_args: int):
    """Benchmark overhead when using pool map func (only sending not receiving)"""
    pool_args = [(i, ) for i in range(n_args)]
    timer = Timer("ms").start()
    with Pool() as p:
        p.map(zero_computing_func, pool_args)
    logger.info("Default overhead: %s", timer.lap().get_total_time())

    default_time = timer.total_time
    args = get_args()
    timer.reset()

    with Pool() as p:
        result = p.map(zero_computing_func, args)
    logger.info("Default overhead + serialize input takes: %s", timer.lap().get_total_time())
    logger.info("=> Overhead of sending input: %s ms", round((timer.total_time - default_time) * 1000, 4))


def benchmark_overhead_multiprocess_map_time(get_args: Callable[[], List[Tuple]], n_args: int):
    """Benchmark overhead when using multiprocessing to simulate pool map (fork)"""
    foo_args = [(i, ) for i in range(n_args)]
    n_cpu = os.cpu_count()

    timer = Timer("ms").start()
    ps = [Process(target=zero_computing_func, args=(foo_args[start:end], )) for start, end in get_batchs(n_args, n_cpu)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    logger.info("Default overhead: %s", timer.lap().get_total_time())

    default_time = timer.total_time
    args = get_args()

    timer.reset()
    ps = [Process(target=zero_computing_func, args=(args[start:end], )) for start, end in get_batchs(n_args, n_cpu)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    logger.info("Default overhead + serialize input takes: %s", timer.lap().get_total_time())
    logger.info("=> Overhead of sending input: %s ms", round((timer.total_time - default_time) * 1000, 4))


def sequential_map(func, args, n_process=None, unpack=False):
    if unpack:
        return [func(*arg) for arg in args]
    else:
        return [func(arg) for arg in args]


def parallel_map_unpack_func(func, queue: Queue, idx: int, batch_args: List[Tuple]) -> None:
    queue.put((idx, [func(*args) for args in batch_args]))


def profiled_parallel_map_unpack_func(func, queue: Queue, idx: int, batch_args: List[Tuple]) -> None:
    begin = time.time()
    result = [func(*args) for args in batch_args]
    exec_time = time.time() - begin
    queue.put((idx, exec_time, result))


def parallel_map(func, args, n_process=None, unpack=False):
    if n_process is None:
        n_process = os.cpu_count()

    queue = Queue()
    batch_args = [args[start:end] for start, end in get_batchs(len(args), n_process)]
    if unpack:
        processes = [
            Process(target=parallel_map_unpack_func, args=(
                func,
                queue,
                idx,
                batch_arg,
            )) for idx, batch_arg in enumerate(batch_args)
        ]
    else:
        assert False

    for p in processes:
        p.start()

    results = {}
    for i in range(n_process):
        idx, res_array = queue.get()
        results[idx] = res_array

    for p in processes:
        p.join()

    return [el for idx, res_array in sorted(results.items(), key=lambda x: x[0]) for el in res_array]


def parallel_pool_map(func, args, n_process=None):
    if n_process is None:
        n_process = os.cpu_count()

    with Pool(n_process) as p:
        return p.map(func, args)


def profiled_parallel_map(func, args, n_process=None, time_unit: str = "ms", unpack=False):
    if n_process is None:
        n_process = os.cpu_count()

    queue = Queue()
    batch_args = [args[start:end] for start, end in get_batchs(len(args), n_process)]

    timer = Timer(time_unit).start()
    if unpack:
        processes = [
            Process(target=profiled_parallel_map_unpack_func, args=(
                func,
                queue,
                idx,
                batch_arg,
            )) for idx, batch_arg in enumerate(batch_args)
        ]
    else:
        assert False

    for p in processes:
        p.start()
    logger.info("Start process takes: %s", timer.lap().get_total_time())

    print(len(processes))
    total_exec_time = 0
    results = {}
    for i in range(n_process):
        idx, exec_time, res_array = queue.get()
        results[idx] = res_array
        total_exec_time += exec_time

    for p in processes:
        p.join()

    result = [el for idx, res_array in sorted(results.items(), key=lambda x: x[0]) for el in res_array]

    logger.info("Total mutli-process time: %s", timer.lap().get_total_time())
    total_exec_time = round(total_exec_time * timer.time_unit / n_process, 4)
    logger.info("Total computing-time: %s %s", total_exec_time, time_unit)
    logger.info("=> Overhead: %s %s", timer.total_time * timer.time_unit - total_exec_time, time_unit)

    return result


def profiled_parallel_pool_map(func, args, n_process=None, time_unit: str = "ms"):
    if n_process is None:
        n_process = os.cpu_count()

    timer = Timer(time_unit).start()
    with Pool(n_process) as p:
        results = p.map(func, args)

    logger.info("Total time: %s", timer.lap().get_total_time())
    return results


T = TypeVar('T')

class AsyncResult(Generic[T]):

    def __init__(self, val: T):
        self.val: T = val

    def get(self) -> T:
        return self.val


class FakePool(object):

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def apply_async(self, func, args):
        return AsyncResult(func(*args))


def get_pool(n_process: int):
    if n_process == 1:
        return FakePool()

    return Pool(n_process)

def func(x, arrays):
    result = 0
    for i in arrays:
        result += x + i
    return result


def func2(args):
    x, arrays = args
    result = 0
    for i in arrays:
        result += x + i
    return result


if __name__ == '__main__':
    set_start_method("fork")
    logger.info("Start method: %s", get_start_method())
    arrays = [i for i in range(5000)]

    # create_args = lambda: ([{"number": i} for i in range(5000000)],)
    # create_args = lambda: [(i, arrays) for i in arrays]
    # create_args = lambda: [(i, list(range(5000))) for i in range(5000)]
    create_args = lambda: [i for i in range(10000000)]
    # benchmark_overhead_time(create_args)
    benchmark_sending_time(create_args)
    # benchmark_overhead_multiprocess_map_time(create_args, n_args=len(arrays))
    # benchmark_overhead_pool_map_time(create_args, n_args=len(arrays))

    # test execution
    # timer = Timer("ms").start()
    # n_iter = 5
    # for i in range(n_iter):
    #     parallel_map(func, create_args())
    #     timer.lap()
    # logger.info("Multi-process map takes: %s", timer.get_average_time())

    # timer.reset()
    # for i in range(n_iter):
    #     profiled_parallel_map(func, create_args())
    #     timer.lap()
    # logger.info("Multi-process map takes: %s", timer.get_average_time())

    # timer.reset()
    # parallel_map(func, create_args())
    # logger.info("Multi-process map takes: %s", timer.lap().get_total_time())

    # timer.reset()
    # for i in range(n_iter):
    #     parallel_pool_map(func2, create_args())
    #     timer.lap()
    # logger.info("Pool map takes: %s", timer.get_average_time())
    #
    # timer.reset()
    # for i in range(n_iter):
    #     list(map(func2, create_args()))
    #     timer.lap()
    # logger.info("Sequential map takes: %s", timer.get_average_time())
