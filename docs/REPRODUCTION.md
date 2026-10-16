# Reproducir sin mezclar la entrega y una ejecución nueva

El repositorio incluye reportes de la ejecución entregada, pero ignora los datos procesados, modelos binarios y base MLflow local. Los programas no deben reutilizar un resumen de la entrega cuando esos artefactos faltan. Para entrenar desde cero, usar un **clon nuevo** y apartar primero los reportes entregados. No ejecutar estos pasos sobre una carpeta que ya contiene una ejecución que se desea continuar.

Desde la raíz de ese clon nuevo, en PowerShell:

```powershell
if (Test-Path -LiteralPath reports-delivered) { throw 'Ya existe reports-delivered; revisar antes de continuar.' }
Rename-Item -LiteralPath reports -NewName reports-delivered
New-Item -ItemType Directory -Path reports
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-stage4.txt
Copy-Item .env.example .env
```

Los reportes originales quedan conservados en `reports-delivered`. Configurar en `.env` la ruta `PAYSIM_CSV` al CSV completo y `MLFLOW_BACKEND=local`. Después seguir los comandos `prepare`, `baseline`, `optimize`, `candidate`, `evaluate`, `explain` y `monitor` del README. Esta ejecución produce sus propias evidencias y runs locales; no debe publicarse como si fueran los identificadores de la entrega original.

Los nuevos hashes de serialización, identificadores MLflow y tiempos pueden diferir. No se deben copiar hashes antiguos para forzar una coincidencia. Los comandos detectan artefactos incompletos o incompatibles y piden conservarlos antes de repetir, en lugar de sobrescribirlos silenciosamente.

Para **consultar** la entrega, no apartar `reports`: abrir los cuatro notebooks ejecutados y HTML incluidos o ejecutar `python -m streamlit run app.py`. La app solo lee reportes y no necesita CSV ni autenticación. Elegir el kernel de Python del entorno instalado al abrir notebooks en Jupyter/VS Code.

Para **continuar** una ejecución local existente, conservar sus `reports`, `artifacts`, `data/processed` y base local correspondientes; no usar la preparación de clon nuevo.
