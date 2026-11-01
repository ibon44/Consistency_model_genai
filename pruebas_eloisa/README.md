# Pruebas de consistency training

CT v2: la implementación está en `system/training/consistency_training_v2.py`.
El CT v1 original sigue en `system/training/consistency_training.py` y guarda en
`checkpoints/`.

Ejecuta los comandos desde la raíz del repositorio.

Empezar:

```powershell
.venv\Scripts\python.exe system\training\consistency_training_v2.py
```

Parar: pulsa Ctrl+C una sola vez y espera al mensaje `Interrupted: saved at iteration ...`.

Continuar:

```powershell
.venv\Scripts\python.exe system\training\consistency_training_v2.py --resume
```

Archivos:

- `mnist_all_cm_ve_ct.pth`: modelo para muestrear.
- `mnist_all_cm_ve_ct_resume.pth`: estado completo para continuar.
