# Pruebas de consistency training

Ejecuta los comandos desde la raíz del repositorio.

Empezar:

```powershell
.venv\Scripts\python.exe system\training\consistency_training.py --checkpoint-dir pruebas_eloisa
```

Parar: pulsa Ctrl+C una sola vez y espera al mensaje `Interrupted: saved at iteration ...`.

Continuar:

```powershell
.venv\Scripts\python.exe system\training\consistency_training.py --checkpoint-dir pruebas_eloisa --resume
```

Archivos:

- `mnist_all_cm_ve_ct.pth`: modelo para muestrear.
- `mnist_all_cm_ve_ct_resume.pth`: estado completo para continuar.
