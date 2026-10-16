# Instalación, configuración y ejecución

El proyecto permite consultar los resultados incluidos o ejecutar el análisis completo. La aplicación local lee los reportes sin requerir el CSV ni credenciales.

## Consulta los resultados incluidos

1. Clona el repositorio y entra en su carpeta.
2. Crea un entorno con Python 3.12.
3. Instala las dependencias y ejecuta el panel.

```powershell
git clone https://github.com/jorgesialer/Proyecto-mle-fraude.git
cd Proyecto-mle-fraude
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abre la dirección local que indica Streamlit. Selecciona un periodo, consulta las métricas y revisa las pestañas de calidad, explicaciones y reportes. Descarga los HTML para abrirlos como páginas en el navegador. Conserva la carpeta `reports`: contiene los resultados que utiliza el panel.

## Configura una ejecución desde cero

Utiliza un clon nuevo para mantener separadas la entrega original y la nueva ejecución. Aparta sus reportes antes de entrenar; no apliques este paso a una ejecución local que quieras continuar.

```powershell
if (Test-Path -LiteralPath reports-delivered) { throw 'Ya existe reports-delivered; revisa su contenido antes de continuar.' }
Rename-Item -LiteralPath reports -NewName reports-delivered
New-Item -ItemType Directory -Path reports
Copy-Item .env.example .env
```

Configura `PAYSIM_CSV` con la ruta al CSV completo y establece `MLFLOW_BACKEND=local` en `.env`. Mantén las credenciales fuera de Git. El archivo `reports-delivered/environment-stage4.json` contiene las versiones de la ejecución original.

## Ejecuta el análisis completo

Ejecuta los módulos desde la raíz y respeta este orden para mantener la separación entre entrenamiento, selección del umbral, evaluación y monitoreo:

```powershell
.\.venv\Scripts\python.exe -m src.fraud.prepare
.\.venv\Scripts\python.exe -m src.fraud.baseline
.\.venv\Scripts\python.exe -m src.fraud.optimize --trials 10 --threads 4
.\.venv\Scripts\python.exe -m src.fraud.candidate
.\.venv\Scripts\python.exe -m src.fraud.evaluate
.\.venv\Scripts\python.exe -m src.fraud.explain
.\.venv\Scripts\python.exe -m src.fraud.monitor
```

La ejecución genera sus propios artefactos, identificadores MLflow y tiempos. Conserva esos resultados juntos; no mezcles sus resúmenes con modelos de otra ejecución. Los comandos comprueban la compatibilidad de datos y modelos antes de reutilizar resultados.

## Abre los notebooks y ejecuta las pruebas

Selecciona el kernel del entorno instalado en Jupyter o VS Code. Consulta los cuatro notebooks en orden. Sus salidas incluidas permiten revisar la entrega sin ejecutar nuevamente el entrenamiento.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Para continuar una ejecución existente, conserva sus carpetas `reports`, `artifacts`, `data/processed` y su base MLflow local. Evita el paso de preparación del clon nuevo.
