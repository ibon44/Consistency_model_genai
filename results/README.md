# Resultados de la comparativa

Todo lo de esta carpeta lo genera `system/evaluation/` desde la raíz del repo
(las figuras de la memoria se leen de aquí, `memoria/memoria_final.tex`).
Los resultados de `memoria/memoria_final_v2.tex`, con CT v2 e iCT v2, están en
[`memoria_final_v2/`](memoria_final_v2/README.md).

| Archivo | Qué es | Cómo se genera |
|---|---|---|
| `tabla_resultados.md` / `.csv` | FD-MNIST, % reconocible, entropía, ms/imagen y aceleración de cada modelo y número de pasos | `compare.py` + `reuse_timing.py` |
| `resultados.json` | Lo anterior más el reparto por dígito de cada configuración | `compare.py` + `reuse_timing.py` |
| `calidad_vs_nfe.png` | FD-MNIST y % reconocible frente a evaluaciones de la red | `compare.py` |
| `distribucion_digitos.png` | Mapa de calor de los dígitos que predice el clasificador | `compare.py` |
| `muestras_ve.png`, `muestras_rf.png` | Muestras de las familias VE y RF: 10 filas (mismo ruido por fila), una columna por modelo y nº de pasos | `compare.py` |
| `muestras_ve_3filas.png`, `muestras_rf_3filas.png` | Versión compacta (3 filas, columnas elegidas en `GRID_3ROWS`) que usa la memoria | `compare.py` |
| `tiempo_vs_n.png` | Tiempo de generación frente al número de imágenes | `reuse_timing.py` |
| `tiempos_referencia.json` | Tiempos medidos una vez con la CPU libre; el tiempo solo depende de la red y del nº de evaluaciones, así que se comparten entre modelos del mismo tipo | medido una vez |
| `variabilidad_ruido.md` / `.csv` | Media y desviación de las métricas de los consistency models con 5 semillas de ruido (mismo modelo entrenado) | `seed_variance.py` |

Orden para regenerarlo todo:

```
python system/evaluation/compare.py            # ~40 min en CPU
python system/evaluation/reuse_timing.py       # sustituye los tiempos por los de referencia
python system/evaluation/seed_variance.py      # ~20 min
```

Los consistency models se comparan con el teacher del que se destilaron: CM-VE
(CD) con el teacher VE crudo (`VE teacher (raw)`), CM-RF con el 1-RF y CM-RF
(reflow) con el 2-RF. CT e iCT no tienen teacher. `VE teacher (EMA)` es el mejor
teacher VE, pero no es el que se destiló.

Las figuras de ablaciones de RF (EMA frente a crudo, 1-RF frente a 2-RF, barrido
de tau) están en `figures/`.
