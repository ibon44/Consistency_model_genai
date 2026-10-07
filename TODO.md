# TODO — Consistency Models (difusión VE + Rectified Flow)

Lista de tareas en el orden en que tienen sentido, con sus dependencias.
Referencia de fechas: plan de trabajo del proyecto (21 sep – 31 oct).

## Fase 0 — Base (ya hecho)

- [x] Arquitectura del repo: paquetes `flow_matching/`, `consistency/`,
      `evaluation/`, y ampliaciones de `models/`, `sampling/`, `training/`.
- [x] Refactor de `models/score_unet.py` para admitir `marginal_prob_std=None`.
- [x] Limpieza de la copia duplicada de MNIST en `consistency_model_iagen/data/`.
- [x] Maestro de difusión (VE) y su sampler (`diffusion/ve.py`,
      `sampling/probability_flow.py`) — trabajo previo al inicio de esta lista.

## Fase 1 — Piezas independientes (sin dependencias entre sí, se pueden repartir en paralelo)

- [x] **1. `flow_matching/rectified_flow.py`** — implementar `interpolate`,
      `target_velocity`, `sigma_t`, `loss_function`.
      *Depende de:* nada (solo del andamiaje ya creado).
- [x] **2. `models/velocity_unet.py`** — implementar `VelocityNet`
      (reutilizar el backbone de `ScoreNet` con `marginal_prob_std=None`).
      *Depende de:* nada.
- [ ] **3. `evaluation/classifier.py`** — implementar `MNISTClassifier`
      (CNN pequeña).
      *Depende de:* nada.
- [ ] **4. `consistency/schedules.py`** — implementar `DiscretizationSchedule`
      y `EMADecaySchedule`.
      *Depende de:* nada.
- [ ] **5. `models/consistency_wrapper.py`** — implementar `c_skip`, `c_out`,
      `forward` (condición de contorno f(x,ε)=x).
      *Depende de:* nada (solo necesita una `sigma_t` que pasar; puede
      probarse ya con `diffusion.ve.VarianceExplodingDiffusion.sigma_t`).

## Fase 2 — Entrenamiento de maestros (Hito 1: 4 de octubre)

- [ ] **6. Entrenar maestro de difusión (VE)** — ejecutar
      `training/ve_training.py` sobre todo MNIST → `checkpoints/mnist_all_ve.pth`.
      *Depende de:* nada (script ya existe y funciona).
- [x] **7. `training/rectified_flow_training.py`** — completar el bucle de
      entrenamiento.
      *Depende de:* 1, 2.
- [x] **8. Entrenar maestro Rectified Flow** — ejecutar el script anterior
      → `checkpoints/mnist_all_rf.pth`.
      *Depende de:* 7.
- [ ] **9. `training/classifier_training.py`** — completar el bucle y
      entrenar → `checkpoints/mnist_classifier.pth`.
      *Depende de:* 3.

## Fase 3 — Destilación a modelos de consistencia (Semana 3: 5–11 oct)

- [ ] **10. `consistency/teachers.py` — `VETeacherAdapter.ode_step`**
      (paso determinista del PF-ODE de la difusión VE).
      *Depende de:* 6 (para poder probarlo con un maestro real), 5.
- [ ] **11. `consistency/teachers.py` — `RectifiedFlowTeacherAdapter.ode_step`**
      (paso determinista del ODE de RF).
      *Depende de:* 1, 8, 5.
- [ ] **12. `consistency/distillation.py` — `consistency_distillation_loss`**
      (Algoritmo 2, agnóstico al maestro).
      *Depende de:* 4, 5, 10, 11.
- [ ] **13. `training/consistency_distillation_training.py`** — completar
      el bucle (carga de checkpoint del maestro, modelo online + EMA,
      optimizador, actualización EMA).
      *Depende de:* 12.
- [ ] **14. Entrenar CM-VE** (`--teacher ve`) → `checkpoints/mnist_all_cm_ve.pth`.
      *Depende de:* 6, 13.
- [ ] **15. Entrenar CM-RF** (`--teacher rf`) → `checkpoints/mnist_all_cm_rf.pth`.
      *Depende de:* 8, 13.

## Fase 4 — Comparativa (Semana 4: 12–18 oct, Hito 2: 18 oct)

- [ ] **16. `sampling/consistency_sampler.py`** — implementar `sample`
      (1-step y multistep, Algoritmo 1).
      *Depende de:* 5.
- [x] **17. `sampling/rectified_flow_sampler.py`** — implementar `sample`
      (integración Euler del ODE).
      *Depende de:* 1.
- [ ] **18. `evaluation/metrics.py`** — implementar `classification_accuracy`,
      `label_distribution`, `label_entropy`.
      *Depende de:* 9.
- [ ] **19. `evaluation/timing.py`** — implementar `benchmark_generation_time`.
      *Depende de:* nada (genérico), pero se usa con 16/17/`probability_flow.py`.
- [ ] **20. `evaluation/compare.py`** — implementar `main`, orquestando
      difusión vs RF vs CM-VE vs CM-RF, guardar tablas/figuras en `results/`.
      *Depende de:* 14, 15, 16, 17, 18, 19.
- [ ] **21. Ejecutar la comparativa completa** y revisar resultados (calidad
      a 1/2/4 pasos, tiempo vs N, inspección visual).
      *Depende de:* 20.

## Fase 5 — Análisis y redacción (Semana 6: 26–31 oct)

- [ ] **22. Redactar el análisis de resultados** (objetivo 4: aceleración,
      calidad vs nº de pasos, diversidad, limitaciones técnicas).
      *Depende de:* 21.
- [ ] **23. Limpieza del repo + preparación del informe/presentación.**
      *Depende de:* 22.

## Extensiones (Semana 5: 19–25 oct, solo si sobra tiempo — en este orden)

- [ ] **E1. `consistency/ct_loss.py` + `training/consistency_training.py`**
      (consistency training sin maestro) y comparación frente a CM-VE/CM-RF.
      *Depende de:* 4, 5, 21 (para comparar contra algo ya medido).
- [ ] **E2. Adaptar arquitectura a color y entrenar sobre CIFAR-10**
      (`models/score_unet_color.py` ya existe como punto de partida).
      *Depende de:* E1 completada (o descartada por falta de tiempo), 21.
- [ ] **E3. Modelo de consistencia condicional** (generar un dígito concreto).
      *Depende de:* E2 completada (o descartada), 21.
