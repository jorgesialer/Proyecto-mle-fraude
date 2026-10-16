# Umbral y evaluación final

El candidato se eligió exclusivamente mediante validación temporal en steps 1–250 y se ajustó en ese mismo periodo. Se mantienen sus parámetros y coeficientes.

Antes de consultar la validación de umbral se establece esta política: maximizar F2 sobre los scores de steps 251–300; ante empate numérico de F2 a 1e-12, elegir el umbral más alto. La comparación es `score >= threshold`. F2 prioriza recall; no representa costos monetarios observados. Los scores ponderados no se declaran calibrados.

La política se guarda en `reports/evaluation/decision_policy.json`, con hashes de modelo y datos. Solo después se abre el test offline 301–400. No se cambia el modelo ni el umbral a partir de sus resultados. Las métricas con umbral 0,5 se incluyen como diagnóstico de referencia, no para volver a elegir.

La inferencia se ejecuta por lotes y guarda predicciones y manifiestos de hash en artifacts. Rehacer gráficos a partir de esas predicciones no constituye una nueva selección; el programa rechaza datos/modelos diferentes en la misma evidencia. Las repeticiones del CLI preservan el resultado existente y no repiten búsquedas sobre el test.

Se guarda un modelo de decisión que combina el pipeline elegido y el umbral congelado. Se verifican las predicciones después de recargarlo. Su `predict` aplica el umbral; `predict_proba` conserva los scores originales. No se vuelve a entrenar con validación ni test, y producción desde 401 permanece excluida de esta etapa.

Referencia: [precision_recall_curve de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html). Su último punto no tiene umbral asociado y se excluye de la optimización.
