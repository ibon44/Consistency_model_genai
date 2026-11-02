# Consistency Models on MNIST

Small-scale reproduction of *Consistency Models* (Song et al., 2023) for the
Generative AI course at Universidad Autónoma de Madrid. We train two teachers
(a Variance Exploding diffusion model and a Rectified Flow model), distill each
one into a consistency model, train consistency models from scratch (CT and
improved CT), and compare all of them in sample quality, diversity and sampling
time.

- **Report:** [`memoria/memoria.pdf`](memoria/memoria.pdf) (NeurIPS template, source in `memoria/memoria.tex`)
- **Tutorial notebook:** [`notebook/tutorial.ipynb`](notebook/tutorial.ipynb)
- **Results (tables and figures of the report):** [`results/`](results/README.md)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

MNIST is in `data/` (torchvision downloads it there if it is missing), and the
trained weights are in `checkpoints/`, so nothing needs to be trained to run
the notebook or the evaluation.

**Run every script from the root of the repository**: the paths to `data/`,
`checkpoints/` and `results/` are relative to it.

## Repository structure

```
system/
  diffusion/        VE diffusion process
  flow_matching/    Rectified Flow
  consistency/      CD, CT and iCT losses, discretization and EMA schedules
  models/           U-Nets (score, velocity) and the consistency-model wrapper
  sampling/         samplers: Euler-Maruyama, predictor-corrector, probability flow,
                    Rectified Flow and consistency sampling
  training/         one training script per model
  evaluation/       MNIST classifier, FD-MNIST and other metrics, timing, comparison
  tests/            sampling demos and the RF ablations of figures/
checkpoints/        trained weights (teachers, consistency models, classifier)
results/            outputs of system/evaluation/, read by the report
figures/            RF ablation figures (EMA vs raw, 1-RF vs 2-RF, tau sweep)
memoria/            report (LaTeX source and PDF)
notebook/           tutorial notebook
```

## Training

Each model has its own script in `system/training/`; the hyperparameters are
constants at the top of each file and some of them can be changed from the
command line (`--help`).

| Model                                    | Script                                                                                                                      | Checkpoint                                                                     |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| VE diffusion teacher                     | `ve_training.py`                                                                                                          | `mnist_all_ve_ema.pth` (EMA); `mnist_all_ve.pth` (raw, see below)              |
| Rectified Flow teacher (1-RF)            | `rectified_flow_training.py`                                                                                              | `mnist_all_rf.pth`, `mnist_all_rf_ema.pth`                                 |
| 2-RF (reflow)                            | `reflow_training.py`                                                                                                      | `mnist_all_2rf.pth`, `mnist_all_2rf_ema.pth`                               |
| CM-VE, consistency distillation          | `consistency_distillation_training.py --teacher ve --teacher-tag ema --tag ema`                                           | `mnist_all_cm_ve_ema.pth`                                                    |
| CM-VE from the raw teacher (ablation)    | `consistency_distillation_training.py --teacher ve`                                                                       | `mnist_all_cm_ve.pth`                                                        |
| CM-RF, consistency distillation          | `consistency_distillation_training.py --teacher rf`                                                                       | `mnist_all_cm_rf.pth`                                                        |
| CM-RF from the 2-RF                      | `consistency_distillation_training.py --teacher rf --teacher-checkpoint ./checkpoints/mnist_all_2rf_ema.pth --tag reflow` | `mnist_all_cm_rf_reflow.pth`                                                 |
| CM-VE, consistency training (CT v1 / v2) | `consistency_training.py` / `consistency_training_v2.py`                                                                | `mnist_all_cm_ve_ct.pth` / `mnist_all_cm_ve_ct_v2.pth`                     |
| CM-VE, improved CT (iCT v1 / v2)         | `consistency_training_improved.py` (`--tag` names the run)                                                              | `mnist_all_cm_ve_ict_n640.pth` / `mnist_all_cm_ve_ict_40k_ema_lrconst.pth` |
| MNIST classifier (evaluation)            | `classifier_training.py`                                                                                                  | `mnist_classifier.pth`                                                       |

The raw VE teacher (`mnist_all_ve.pth`, only used in the ablation of the report) was
trained with an earlier version of `ve_training.py` (commit `227ce4c`: 40 epochs,
constant learning rate, no EMA). The current script always uses cosine decay.

For example:

```bash
python system/training/consistency_distillation_training.py --teacher rf
```

## Evaluation

`system/evaluation/compare.py` evaluates every model with the same protocol
(FD-MNIST, the Fréchet distance on the features of our MNIST classifier, plus
class diversity and sampling time) and writes the tables and figures to
`results/`:

```bash
python system/evaluation/compare.py --no-timing   # about 40 min on CPU
python system/evaluation/reuse_timing.py          # reference timings
python system/evaluation/seed_variance.py         # noise-seed variability, about 20 min
python system/evaluation/cm_rf_teacher_weights.py # CM-RF from EMA or raw teacher weights
```

`python system/evaluation/compare.py --quick --results-dir ./results_quick` checks
in about a minute that everything runs. Keep `--results-dir`: otherwise the quick
run overwrites `results/`, which the report reads. See [`results/README.md`](results/README.md) for what each
output file contains.

## Authors

- Ibón de Mingo Arroyo ([@ibon44](https://github.com/ibon44))
- Eloísa Denia García ([@Eloodenia](https://github.com/Eloodenia))
- Matin Esmaeilzadeh ([@matinesm](https://github.com/matinesm))
