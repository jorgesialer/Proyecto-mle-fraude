# Datos

Fuente: dataset entregado para MLE III, también disponible en:

- https://drive.google.com/drive/folders/1pyzE2Fv6_TuxidmJ5gHv6mAFy9nJ-c28
- https://www.kaggle.com/datasets/charlesbeauchamp/ps-20174392719-1491204439457-logcsv

Colocar el CSV completo en `data/raw/` o configurar `PAYSIM_CSV` en el `.env` local. El original no se altera ni se copia a Git; pesa aproximadamente 494 MB.

`python -m src.fraud.prepare` genera cuatro Parquet en `data/processed/`. Se conservan todas las filas, sin muestreo o balanceo. El manifiesto en `reports/dataset_profile.json` registra hashes SHA-256 y conteos. Los archivos grandes y `.env` están excluidos de Git.

`sample_illustrative.csv` contiene solo las primeras 50 filas de cada partición, para ilustrar el esquema. Es una muestra no representativa y no se usa para entrenar o evaluar.

La preparación detiene el proceso ante categorías desconocidas, target no binario, nulos originales, steps inválidos o valores monetarios negativos/no finitos. Los NaN del ratio monto/saldo son derivados e intencionales. El alcance de estos controles se documenta en el diccionario.
