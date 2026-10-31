# Diario del proyecto

Registro breve de lo que se ha hecho, entrada más reciente arriba.

---

## 2026-10-31 — Consistency training mejorado y reevaluación completa

- Nuevo `consistency/ict_loss.py` + `training/consistency_training_improved.py`:
  consistency training con las técnicas de Song & Dhariwal (2024) (sin target EMA,
  curriculum exponencial de N, ruido lognormal y pérdida ponderada por
  1/(σ_{n+1}-σ_n)). Dos fases: 20 000 iteraciones (N 11→161) y 32 289 más desde
  esos pesos (batch 128, N 161→641). Resultado: mejor que CD a 2 y 4 pasos
  (FD 274 y 179 frente a 388 y 321) con unas 2 veces más imágenes de entrenamiento.
- `training/consistency_training.py`: guardado atómico, `_resume` final, aviso para
  no pisar un run, LR correcto al reanudar y guardado con Ctrl+C.
- `evaluation/compare.py` compara cada CM con el teacher del que se destiló (VE
  crudo, 1-RF, 2-RF) e incluye CT, iCT (dos fases) y las figuras de 3 filas de la
  memoria. Nuevos `reuse_timing.py` (tiempos de referencia por arquitectura) y
  `seed_variance.py` (variabilidad por muestreo). Todo en `results/` (ver su README).
- Hallazgo: relativo a su propio teacher, CM-VE es mejor copia (1,6× el FD del VE
  crudo a 4 pasos) que CM-RF (3,4× el del 1-RF); CM-RF gana en absoluto porque su
  teacher es mucho mejor (FD 78,5 frente a 206).
- `memoria/memoria_final.tex` actualizada a esta ejecución (cuerpo en 8 páginas).

---

## 2026-09-27 — Maestro Rectified Flow implementado y entrenado

- Implementadas las tareas 1, 2, 7, 8 y 17 de `TODO.md`, siguiendo
  Liu et al., 2022 (*Flow Straight and Fast*):
  - `flow_matching/rectified_flow.py`: `interpolate`
    (`x_t = (1-t)·x_0 + t·x_1`), `target_velocity` (`x_1 - x_0`),
    `sigma_t` (`t·sigma_noise`) y `loss_function` (MSE de la
    velocidad, media sobre batch y píxeles, igual que el Algoritmo 2
    del paper).
  - `models/velocity_unet.py`: `VelocityNet` envuelve una
    `ScoreNet(marginal_prob_std=None)` en `self.net`, así que no
    duplica la UNet. Los pesos del checkpoint llevan el prefijo `net.`:
    hay que cargarlo siempre con `VelocityNet()`.
  - `sampling/rectified_flow_sampler.py`: Euler de `n_steps` pasos
    uniformes de t=1 a t=0. Devuelve `(prior, x_0)` con `x_0` en
    [0, 1], igual que `ProbabilityFlow.sample`.
  - `training/rectified_flow_training.py`: bucle completo, calcado de
    `ve_training.py`.
  - Nuevo `tests/rf_sample_and_plot.py`: genera muestras con
    N = 1, 2, 10, 50 pasos partiendo del mismo ruido en cada fila.
- **Convención de tiempo (distinta a la del paper)**: en el repo
  `x_0` = datos (t=0) y `x_1` = ruido (t=1); el paper usa la contraria.
  *Por qué*: así coincide con la difusión VE y con los modelos de
  consistencia (datos en t≈ε, ruido en t=T). El objetivo de RF es
  simétrico en el tiempo, así que solo cambia el signo: al generar se
  integra hacia atrás, con `dt < 0`.
- **Renombrado `sigma_data` → `sigma_noise`** en `RectifiedFlow` y en
  su script de entrenamiento (`--sigma-noise`). *Por qué*: en RF es la
  desviación típica del **ruido** `x_1`, mientras que `sigma_data` en
  `ConsistencyWrapper` es la de los **datos** (como en Song et al.,
  2023). Con el mismo nombre era fácil pasar uno por otro.
- Entrenado `checkpoints/mnist_all_rf.pth`: todo MNIST, 25 épocas,
  Adam lr=1e-3, CPU (~36 min). Pérdida media por época: 0.28
  (época 2) → 0.196 (época 25). Se estanca porque la pérdida de flow
  matching tiene un mínimo no nulo: la red aprende
  `E[x_1 - x_0 | x_t]`, no cada par concreto.
- Resultados (`results/mnist_all_rf_samples.png`): con N=10 y N=50 los
  dígitos son buenos y casi idénticos (el ODE está resuelto con ~10
  pasos), con fondos limpios y 2-3 fallos de cada 10. Con N=1 sale la
  misma mancha borrosa para cualquier ruido: la *media* de los datos,
  tal como predice el paper para un 1-rectified flow. Es justo lo que
  la destilación a modelo de consistencia debe resolver.
- Pendiente para el equipo de destilación: tarea 11
  (`RectifiedFlowTeacherAdapter.ode_step`, un paso de Euler
  `x_t + (t_prev - t)·v(x_t, t)`). Ojo: `TeacherAdapter` declara `T` y
  `eps`, pero ningún adaptador les da valor.
- Posibles mejoras del maestro, si sobra tiempo: EMA de los pesos
  (el paper la usa) y lr=2e-4. Si se aplican, aplicarlas también al
  maestro VE para que la comparativa sea justa.

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
