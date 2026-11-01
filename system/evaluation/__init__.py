"""
imageGenerativeAI.evaluation

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides the tools used to compare the diffusion and
    Rectified Flow teachers against the consistency models distilled
    from each of them: a reference MNIST classifier, quality/diversity
    metrics, generation-time benchmarking, and an orchestration script
    that ties them together into the project's final report figures.

License:
    Proprietary. All rights reserved.
"""

__all__ = [
    "MNISTClassifier",
]

from .classifier import MNISTClassifier
