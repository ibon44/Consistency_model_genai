# Diario del proyecto

Registro breve de lo que se ha hecho, entrada más reciente arriba.

---

## 2026-09-24 — Arquitectura para Rectified Flow + Consistency Models

- Diseñada y creada la estructura de carpetas para el proyecto completo
  (ver `TODO.md` para el plan de tareas derivado de esta arquitectura):
  - `flow_matching/`: proceso Rectified Flow, como maestro alternativo
    a la difusión VE.
  - `consistency/`: lógica de entrenamiento de los modelos de
    consistencia (schedules N(k)/μ(k), adaptadores de maestro,
    pérdidas de *consistency distillation* y *consistency training*).
  - `evaluation/`: clasificador de referencia, métricas de
    calidad/diversidad y benchmarking de tiempos, más el script que
    orquesta la comparativa final.
  - Ampliaciones en `models/` (backbone de velocidad, wrapper de
    consistencia) y `sampling/` (sampler de RF y de consistencia).
  - Todo el código nuevo está scaffoldeado (firmas, docstrings,
    `NotImplementedError`/`TODO`), no implementado todavía.
- **Cambio en `models/score_unet.py`**: se ha hecho que
  `marginal_prob_std` sea opcional (`None` por defecto ya no se acepta
  como tal, pero pasar `None` explícitamente desactiva el reescalado
  final). *Por qué*: `ScoreNet` divide su salida por
  `marginal_prob_std(t)` porque así se define la parametrización de un
  modelo de *score* (Song et al., 2021). Un modelo de consistencia, en
  cambio, necesita una red "en crudo" (sin ese reescalado) porque es el
  propio `ConsistencyWrapper` (`models/consistency_wrapper.py`) quien
  aplica su parametrización `c_skip(t)·x + c_out(t)·backbone(x,t)` para
  imponer la condición de contorno `f(x,ε)=x`. Sin este cambio habría
  que duplicar toda la arquitectura UNet solo para quitar una línea;
  con él, `ScoreNet(marginal_prob_std=None)` se reutiliza tal cual como
  backbone del modelo de consistencia. El cambio es retrocompatible:
  `ve_training.py` sigue pasando su función `sigma_t` y no se ve
  afectado (verificado).
- Eliminada una copia duplicada de MNIST descargada dentro de
  `consistency_model_iagen/data/MNIST/` — los scripts de entrenamiento
  usan `./data` en la raíz del repo (`CONFIG.data_dir`), así que esa
  copia interna no la usaba nadie y solo inducía a confusión sobre
  dónde vive realmente el dataset.
- Creado `TODO.md` con la lista de tareas ordenada y sus dependencias.

---

## Antes de esta fecha (resumen, no registrado día a día)

- Implementado el proceso de difusión VE (`diffusion/ve.py`) y su
  clase base (`diffusion/base.py`), junto con VP y sub-VP.
- Implementado `ScoreNet` (`models/score_unet.py`) con *random Fourier
  features* para el embedding temporal.
- Implementados los samplers de difusión: Euler-Maruyama,
  predictor-corrector, integrador exponencial y *probability flow* ODE.
- Implementado `training/ve_training.py` y entrenados varios
  checkpoints (`mnist_all_ve.pth`, `mnist_digit4_ve.pth`, etc. en
  `checkpoints/`, logs correspondientes en `logs/`).
- Utilidades base del proyecto: carga de MNIST (`data/`), dispositivo,
  logger, semilla y visualizaciones (`utils/`).
