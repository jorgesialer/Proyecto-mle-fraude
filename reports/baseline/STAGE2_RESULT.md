# Resultado de etapa 2: baseline temporal

Se entrenaron DummyClassifier(prior) y un árbol de decisión de profundidad 6 en tres folds temporales, sin muestreo. Se utilizaron solo steps 1–250. El bloque del umbral, la prueba offline y producción siguen reservados.

## Resultados

- AP media del árbol: 0.994281; desviación entre ventanas: 0.005172.
- Recall combinado: 99.6972%; precision combinada: 69.1306%, con umbral fijo 0,5.
- Fraudes detectados: 1,646; omitidos: 5; falsas alarmas: 735.
- Validación fuera de muestra: 2,188,776 operaciones de tres ventanas disjuntas (steps 101–250).
- La media de AP del predictor trivial es 0.000818; no emite alertas con umbral 0,5.

La media de métricas entre folds no es igual a recalcularlas sobre todas las predicciones juntas. El CSV presenta ambas evidencias por separado. Estas cifras no son resultados de la prueba offline final ni de producción.

## Interpretación

El árbol ordena bien, pero el umbral fijo genera falsas alarmas: recall alto no significa precision perfecta. El peso de clases altera los scores, que no se presentan como probabilidades calibradas. Random Search y Optuna se compararán después; no se garantiza que superen al baseline.

El contrato posterior a transacción incluye saldos finales y discrepancias. En una simulación, esas relaciones pueden separar muy bien los fraudes. El resultado no acredita prevención antes de ejecutar operaciones ni desempeño sobre datos bancarios reales.

## Evidencia y reproducibilidad

- Run local: `fde5ae49c22248ffa6da2c03922729e7`, experimento `fraud-baseline-temporal`.
- Seis runs hijos contienen parámetros y métricas de cada modelo/fold.
- Modelo sklearn con firma, ejemplo de entrada y dependencias; pipeline joblib y modelo MLflow recargados con predicciones idénticas.
- Guardado explícito en cloudpickle para el pipeline creado en esta ejecución. El transformador tiene un módulo estable y no depende del proceso principal.
- La primera exportación falló por el formato por defecto de MLflow; se completó el mismo run con los modelos y métricas ya calculados, sin repetir los entrenamientos. La duración del refit final no se conservó antes del error y se deja sin informar; las duraciones de cada fold sí están registradas.
- Repetir el CLI reutiliza la evidencia existente si coinciden contrato, hash de datos y hash del modelo.

## Conexión remota

Se reutilizó la configuración de DagsHub de la V2 en el .env local, excluido de Git. Tras la autorización explícita del usuario se publicó el modelo y sus evidencias en el experimento separado `fraud-baseline-temporal` (id 1). [Run remoto verificado](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/1/runs/00303d2bd7304de6868196341eff688d), estado FINISHED. Los seis runs hijos están finalizados; las métricas remotas coinciden con las locales y el modelo descargado tiene el mismo hash que el original. `remote_tracking.json` conserva los identificadores y comprobaciones. Los experimentos de aprobación hipotecaria no se modifican. No se enviaron el CSV completo ni credenciales.

Se completaron diez pruebas automatizadas. El segundo notebook se ejecutó sin errores y se exportó como `02_entrenamiento.html`. La figura PR/matriz de confusión se revisó visualmente.

## Próximo trabajo

Optimización con Random Search y Optuna/TPE sobre la misma familia de boosting y folds, seguida de selección de umbral y evaluación offline final. El modelo actual es baseline y no está registrado como champion productivo.
