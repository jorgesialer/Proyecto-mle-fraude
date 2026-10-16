# Model Card: detección de fraude PaySim

## Modelo y uso previsto

XGBoost binario, 90 árboles, profundidad 3, elegido por AP media en validación temporal de desarrollo. Objetivo: isFraud (1=fraude). Caso educativo de clasificación posterior a transacción y reproducción de monitoreo, con revisión de alertas. No se evalúa aprobación crediticia ni riesgo de impago.

El modelo de decisión combina el pipeline de features con el umbral 0.9741922617. El contrato de entrada contiene trece predictores preparados, descritos en `docs/DATA_DICTIONARY.md`. `predict` aplica el umbral y `predict_proba` expone scores no calibrados.

## Datos y entrenamiento

Dataset PaySim del curso, 6.362.620 filas. Entrenamiento y búsqueda: steps 1–250; ajuste del umbral: 251–300; test offline: 301–400; producción simulada desde 401. No se reentrenó con validación o test. Se conservaron todas las filas. Se compararon Random Search y TPE con diez configuraciones y tres folds por método.

Se excluyen target, alerta isFlaggedFraud, identificadores crudos, row_id y step de los predictores. El tipo usa categorías contractuales; valores numéricos float32 y NaN nativo. La ponderación de clase se calcula solo en entrenamiento.

## Evaluación final independiente

Sobre 1,713,380 transacciones de steps 301–400: AP=0.999269, ROC-AUC=1.000000, precision=98.5294%, recall=99.6283%, F2=0.994065. Matriz: TN=1712288, FP=16, FN=4, TP=1072.

El umbral 0,5 obtiene F2 algo mayor en ese test; se conserva el umbral elegido previamente para no seleccionar con datos de evaluación. No se presentan estas métricas como desempeño online real.

## Limitaciones y equidad

Las relaciones contables del simulador y los saldos finales pueden producir resultados muy altos. No son evidencia de ausencia de sobreajuste, causalidad o desempeño en fraude real. El alto umbral no representa una probabilidad calibrada. Los datos no incluyen atributos demográficos protegidos; no se inventan ni se afirma una auditoría de equidad demográfica. El análisis por tipo o monto debe interpretarse como segmentación, no como prueba de fairness.

TreeSHAP global/local, permutación y PDP/ICE completados en una muestra uniforme de 20.000 filas offline (16 fraudes). Las variables principales fueron origin_debit_discrepancy, newbalanceOrig, destination_is_merchant. El error máximo de aditividad fue 4.77e-06. PDP/ICE y permutación pueden crear combinaciones irreales por correlación entre variables originales y derivadas.

En producción histórica (575.590 filas): TP=3.715, FN=21, FP=7, TN=571.847; precisión=99.8119%, recall=99.4379%, AP=0.999668. Una de siete ventanas presentó drift según la política propia; ninguna cruzó los límites de desempeño. ADWIN no emitió señales observadas. Un escenario separado con etiquetas alteradas permitió verificar sensibilidad (demora de 76 filas); no se mezcló con resultados observados. Cambios de prevalencia o distribuciones no prueban por sí solos concept drift. El modelo no debe actualizarse ni cambiar su umbral usando los resultados reservados como criterio de selección.

## Trazabilidad

Hash del candidato: `7733ec68318a19269286f2a543801230d0a44cd021ff62b589c4a9d5661f316e`. Hash de la política: `5f1eb9b9b77b1b15fa15a69a8cd659a1f71d7c16f81914586f5dffab36285d37`. Hash del modelo de decisión: `861c70da70553195a55ba6f4e044b261b39767cc07dd1d0fd1142c62f066d1db`. Evidencia en `reports/evaluation/`, run MLflow local `de732b2a5a23477ba0d67dccb89ef67b`. El artefacto se puede cargar desde `artifacts/evaluation/decision_model.joblib` en el entorno reproducible del proyecto. La publicación de una release y el despliegue de la aplicación siguen pendientes.

## Publicación verificada

[Modelo y evaluación final](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/3/runs/5cc382370d9f41eda4e300af13eb1ba1). Registro `fraud-paysim-posttransaction`, versión 1, READY, etapa Production para replay educativo. No hay despliegue en vivo. El modelo MLflow descargado reproduce exactamente scores y decisiones en 1.000 filas; ambos binarios se verificaron por hash. URI: `runs:/5cc382370d9f41eda4e300af13eb1ba1/model`.
