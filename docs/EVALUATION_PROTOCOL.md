# Umbral y evaluación final

La validación temporal selecciona el candidato en steps 1–250. El ajuste final utiliza ese mismo periodo y fija sus parámetros antes de consultar los bloques posteriores.

La política de decisión maximiza F2 en los steps 251–300. Ante un empate numérico de F2 a 1e-12, el selector elige el umbral más alto. La regla es `score >= threshold`. F2 prioriza recall; no representa costos monetarios observados. Los scores ponderados no constituyen probabilidades calibradas.

El módulo de evaluación guarda la política en `reports/evaluation/decision_policy.json`, junto con los hashes del modelo y los datos, antes de abrir el test 301–400. La evaluación conserva el modelo y el umbral. Las métricas con umbral 0,5 funcionan como diagnóstico de referencia y no intervienen en una nueva selección.

La inferencia procesa lotes y guarda predicciones y manifiestos de hash en `artifacts`. Los gráficos reutilizan esas predicciones. El programa rechaza combinaciones incompatibles de datos y modelos; las ejecuciones posteriores del CLI conservan el resultado existente sin repetir búsquedas sobre el test.

El artefacto de decisión combina el pipeline elegido y el umbral fijo. La comprobación de recarga compara sus predicciones con las del modelo original. `predict` aplica el umbral y `predict_proba` conserva los scores. El entrenamiento excluye validación y test; producción desde 401 permanece fuera de esta etapa.

Referencia: [precision_recall_curve de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html). Su último punto no tiene umbral asociado y se excluye de la optimización.
