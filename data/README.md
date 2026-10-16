# Datos

Usé el archivo `PS_20174392719_1491204439457_log.csv` de PaySim indicado para el proyecto. La fuente pública es [Kaggle](https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv).

El CSV completo pesa aproximadamente 494 MB. Para ejecutar el proyecto, se puede guardar en `data/raw/` o indicar su ruta con `PAYSIM_CSV` en el archivo `.env`.

`python -m src.fraud.prepare` genera las cuatro particiones en `data/processed/`. Conservé todas las filas para entrenar y evaluar. Los conteos y el identificador del archivo están en `reports/dataset_profile.json`.

Incluí `sample_illustrative.csv` para mostrar las columnas sin tener que descargar el archivo completo. Tiene las primeras 50 filas de cada partición; no representa la distribución del dataset y no se utilizó para entrenar. El CSV completo, los Parquet y las credenciales se mantienen fuera de Git.
