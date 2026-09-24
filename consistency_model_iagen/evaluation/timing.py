#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/timing.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Benchmarks generation time for a given sampler (teacher or
    consistency model) as a function of the number of requested
    samples N, as required by objective 3 of the project proposal
    ("se variará el valor de N para estudiar cómo escala el tiempo de
    generación en ambos modelos").

    TODO: implement `benchmark_generation_time`.

License:
    Proprietary. All rights reserved.
"""

from typing import Callable, Dict, List


def benchmark_generation_time(
    generate_fn: Callable[[int], None],
    sample_sizes: List[int],
    n_repeats: int = 3,
) -> Dict[int, float]:
    """Times `generate_fn(n)` for each n in `sample_sizes`.

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
    raise NotImplementedError
