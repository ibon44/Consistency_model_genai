# Resultados de la comparativa

Todo lo de esta carpeta lo genera `system/evaluation/` desde la raíz del repo
(las figuras de la memoria, `memoria/memoria.tex`, se leen de aquí).

| Archivo | Qué es | Cómo se genera |
|---|---|---|
| `tabla_resultados.md` / `.csv` | FD-MNIST, % reconocible, entropía, ms/imagen y aceleración de cada modelo y número de pasos | `compare.py` + `reuse_timing.py` |
| `resultados.json` | Lo anterior más el reparto por dígito de cada configuración | `compare.py` + `reuse_timing.py` |
| `calidad_vs_nfe.png` | FD-MNIST y % reconocible frente a evaluaciones de la red | `compare.py` |
| `distribucion_digitos.png` | Mapa de calor de los dígitos que predice el clasificador | `compare.py` |
| `muestras_ve.png`, `muestras_rf.png` | Muestras de las familias VE y RF: 10 filas (mismo ruido por fila), una columna por modelo y nº de pasos | `compare.py` |
| `muestras_ve_3filas.png`, `muestras_rf_3filas.png` | Versión compacta (3 filas, columnas elegidas en `GRID_3ROWS`) que usa la memoria | `compare.py` |
| `muestras_ve_sin_ema.png` | Teacher VE y CM-VE (CD) con y sin EMA en el teacher (5 filas, columnas en `NOEMA_GRID`); es la ablación del apéndice | `compare.py` |
| `tiempo_vs_n.png` | Tiempo de generación frente al número de imágenes | `reuse_timing.py` |
| `tiempos_referencia.json` | Tiempos medidos una vez con la CPU libre; el tiempo solo depende de la red y del nº de evaluaciones, así que se comparten entre modelos del mismo tipo | medido una vez |
| `variabilidad_ruido.md` / `.csv` | Media y desviación de las métricas de los consistency models con 5 semillas de ruido (mismo modelo entrenado) | `seed_variance.py` |
| `cm_rf_pesos_teacher.md` / `.csv` | CM-RF destilado desde los pesos EMA y desde los pesos crudos del 1-RF | `cm_rf_teacher_weights.py` |

Orden para regenerarlo todo:

```
python system/evaluation/compare.py --no-timing   # ~17 min en CPU
python system/evaluation/reuse_timing.py          # tiempos de referencia
python system/evaluation/seed_variance.py         # ~15 min
python system/evaluation/cm_rf_teacher_weights.py
```

Los consistency models se comparan con el teacher del que se destilaron: CM-VE
(CD) con el `VE teacher` (pesos EMA, `checkpoints/mnist_all_ve_ema.pth`), CM-RF
con el 1-RF y CM-RF (reflow) con el 2-RF. CT e iCT no tienen teacher.

CM-VE (CD) se entrenó con
`python system/training/consistency_distillation_training.py --teacher ve --teacher-tag ema --tag ema`
y se guarda en `checkpoints/mnist_all_cm_ve_ema.pth`.

Ablación (solo apéndice de la memoria): `VE teacher (no EMA)` es el primer teacher VE
(40 épocas, lr constante, sin EMA) y `CM-VE (CD, no-EMA teacher)` el CM-VE que se
destiló de él. Salen en `tabla_resultados` y en `resultados.json`, pero no en las
figuras principales (`calidad_vs_nfe.png`, `muestras_ve*.png`,
`distribucion_digitos.png`) sino en `muestras_ve_sin_ema.png`. `CM-VE (iCT v1)` y
`CM-VE (iCT v1, stage 1)` salen en la tabla completa y en `muestras_ve.png`, pero no
en `calidad_vs_nfe.png`.

Modelos de consistency training (sin teacher):

| Modelo | Checkpoint | Entrenamiento |
|---|---|---|
| CM-VE (CT v1) | `checkpoints/mnist_all_cm_ve_ct.pth` | `system/training/consistency_training.py` |
| CM-VE (CT v2) | `checkpoints/mnist_all_cm_ve_ct_v2.pth` | `system/training/consistency_training_v2.py` |
| CM-VE (iCT v1) | `checkpoints/mnist_all_cm_ve_ict_n640.pth` | `system/training/consistency_training_improved.py` en dos fases |
| CM-VE (iCT v2) | `checkpoints/mnist_all_cm_ve_ict_40k_ema_lrconst.pth` | `consistency_training_improved.py --iterations 40000 --sample-ema 0.9995 --constant-lr` |

`variabilidad_ruido` incluye todos los consistency models (también el CD sin EMA),
medidos en una sola ejecución con las mismas semillas. Si se relanzan `compare.py` o
`seed_variance.py` con otro orden o conjunto de modelos, las filas multipaso cambian un
poco, porque ese muestreo usa el generador global.

Las figuras de ablaciones de RF (EMA frente a crudo, 1-RF frente a 2-RF, barrido
de tau) están en `figures/`.
