#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/timing.py

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Benchmarks generation time for a given sampler (teacher or
    consistency model) as a function of the number of requested
    samples N, as required by objective 3 of the project proposal
    ("se variará el valor de N para estudiar cómo escala el tiempo de
    generación en ambos modelos").

    Follows the usual timing good practices: one warm-up run before
    measuring (the first call is always slower: memory allocation,
    lazy initialization), gradients disabled, several repeats averaged,
    and CUDA synchronization before reading the clock when running on
    a GPU (GPU work is asynchronous).

License:
    Proprietary. All rights reserved.
"""

import time
from typing import Callable, Dict, List

import torch


def _synchronize() -> None:
    """Waits for pending GPU work so the clock measures it (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_generation_time(
    generate_fn: Callable[[int], None],
    sample_sizes: List[int],
    n_repeats: int = 3,
) -> Dict[int, float]:
    """Times `generate_fn(n)` for each n in `sample_sizes`.

    Runs `generate_fn` once with the smallest n as a warm-up (not
    timed), then, for every n, calls it `n_repeats` times and averages
    the wall-clock time measured with time.perf_counter. Everything
    runs under torch.no_grad(), since generation never needs gradients.

    Args:
        generate_fn: Callable that generates `n` samples given `n`
            (e.g. a closure wrapping a sampler's `.sample()` call with
            fixed model/device/step-count, varying only the batch
            size). Should not return the samples themselves.
        sample_sizes: Values of N to benchmark, e.g. [1, 8, 32, 128].
        n_repeats: Number of repeats per N, averaged for stability.

    Returns:
        Mapping from N to average wall-clock generation time (seconds).
    """
    results: Dict[int, float] = {}
    with torch.no_grad():
        generate_fn(min(sample_sizes))
        _synchronize()

        for n in sample_sizes:
            elapsed = []
            for _ in range(n_repeats):
                start = time.perf_counter()
                generate_fn(n)
                _synchronize()
                elapsed.append(time.perf_counter() - start)
            results[n] = sum(elapsed) / len(elapsed)
    return results
