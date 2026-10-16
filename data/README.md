# Datos

El proyecto utiliza el archivo `PS_20174392719_1491204439457_log.csv` de PaySim. La fuente pública es [Kaggle](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv).

El CSV completo pesa aproximadamente 494 MB. Guarda el archivo en `data/raw/` o configura su ruta mediante `PAYSIM_CSV` en `.env`.

`python -m src.fraud.prepare` genera las cuatro particiones en `data/processed/`. La preparación conserva todas las filas para entrenar y evaluar. Los conteos y el identificador del archivo están en `reports/dataset_profile.json`.

El archivo `sample_illustrative.csv` muestra las columnas sin requerir la descarga del dataset completo. Tiene las primeras 50 filas de cada partición; no representa la distribución del dataset y no se utilizó para entrenar. El CSV completo, los Parquet y las credenciales se mantienen fuera de Git.
