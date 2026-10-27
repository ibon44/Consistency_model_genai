#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/compare_cm_rf_reflow.py

Authors:
    Adam Maltoni <admsit0>
    Ibón de Mingo Arroyo <ibon44>

Description:
    Visual inspection of Reflow (`training/reflow_training.py`): the
    consistency model distilled from the 1-Rectified Flow teacher vs
    the one distilled from the 2-Rectified Flow, plus both teachers at
    1 and 100 Euler steps. The 1-step teacher row is where Reflow
    should show most (straighter paths -> one Euler step already lands
    near a digit instead of the blurry data mean). Same layout and
    same initial noise per cell as `tests/compare_cm_rf_ema_noema.py`.

License:
    Proprietary. All rights reserved.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tests.compare_cm_rf_ema_noema as panels


# ============================== CONFIG ==============================
# label -> (consistency model checkpoint, teacher it was distilled from)
VARIANTS = {
    "1-RF": (
        "./checkpoints_rf/mnist_all_cm_rf.pth",
        "./checkpoints_rf/mnist_all_rf_ema.pth",
    ),
    "2-RF (reflow)": (
        "./checkpoints_rf/mnist_all_cm_rf_reflow.pth",
        "./checkpoints_rf/mnist_all_2rf_ema.pth",
    ),
}
# ======================================================================


if __name__ == "__main__":
    panels.TEACHER_STEP_COUNTS = [1, 100]
    panels.compare(VARIANTS, "cm_rf_1rf_vs_reflow")
