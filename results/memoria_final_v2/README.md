# Resultados de la memoria final v2

Figuras y tablas de `memoria/memoria_final_v2.tex`. Los archivos son los mismos que
describe [`results/README.md`](../README.md), con estos modelos añadidos o renombrados:

| Modelo | Checkpoint | Entrenamiento |
|---|---|---|
| CM-VE (CT v1), antes CT | `checkpoints/mnist_all_cm_ve_ct.pth` | `system/training/consistency_training.py` |
| CM-VE (CT v2) | `checkpoints/mnist_all_cm_ve_ct_v2.pth` | `system/training/consistency_training_v2.py` |
| CM-VE (iCT v1), antes iCT | `checkpoints/mnist_all_cm_ve_ict_n640.pth` | `system/training/consistency_training_improved.py` en dos fases |
| CM-VE (iCT v2) | `checkpoints/mnist_all_cm_ve_ict_40k_ema_lrconst.pth` | `consistency_training_improved.py --iterations 40000 --sample-ema 0.9995 --constant-lr` |

Para regenerarlos, desde la raíz del repo:

```powershell
.venv\Scripts\python.exe system\evaluation\compare.py --no-timing
.venv\Scripts\python.exe system\evaluation\reuse_timing.py
```

`cm_rf_pesos_teacher.md` y `.csv` comparan CM-RF destilado desde los pesos EMA y crudos
del 1-RF; se generan con `system/evaluation/cm_rf_teacher_weights.py`.

En `variabilidad_ruido`, las filas de CT v2 e iCT v2 se midieron aparte, con las mismas
semillas, para no alterar las demás. Si se relanza `seed_variance.py`, las filas
multipaso cambiarán un poco, porque ese muestreo usa el generador global.
