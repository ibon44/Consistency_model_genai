"""
imageGenerativeAI.consistency

Authors:
    Matin Esmaeilzadeh <matinesm>
    Eloísa Denia García <Eloodenia>
    Ibón de Mingo Arroyo <ibon44>

Description:
    This module provides the training logic for consistency models
    (Song et al., 2023): discretization/EMA schedules, teacher adapters
    that make a diffusion process (VE) or a flow-matching process
    (Rectified Flow) interchangeable as a distillation teacher, and the
    consistency distillation (CD) and consistency training (CT) losses.

License:
    Proprietary. All rights reserved.
"""

__all__ = [
    "ConstantSchedule",
    "DiscretizationSchedule",
    "EMADecaySchedule",
    "karras_sigma_time_steps",
    "karras_time_steps",
    "squared_l2_distance",
    "pseudo_huber_distance",
    "TeacherAdapter",
    "VETeacherAdapter",
    "RectifiedFlowTeacherAdapter",
    "consistency_distillation_loss",
    "consistency_training_loss",
]

from .ct_loss import consistency_training_loss
from .distances import pseudo_huber_distance, squared_l2_distance
from .distillation import consistency_distillation_loss
from .schedules import (
    ConstantSchedule,
    DiscretizationSchedule,
    EMADecaySchedule,
    karras_sigma_time_steps,
    karras_time_steps,
)
from .teachers import (
    RectifiedFlowTeacherAdapter,
    TeacherAdapter,
    VETeacherAdapter,
)
