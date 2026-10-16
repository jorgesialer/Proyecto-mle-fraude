# Umbral y evaluación offline final

Se mantuvo el XGBoost de Random Search, entrenado en steps 1–250, sin modificar coeficientes ni hiperparámetros. El umbral se eligió en las 825,358 operaciones de steps 251–300, maximizando F2. Ante empate, se eligió el umbral más alto.

**Umbral congelado:** 0.9741922617, con regla score >= umbral. Es un score ponderado, no una probabilidad calibrada del 97,4 %.

## Resultados por periodo

| Periodo | Operaciones | Precision | Recall | F2 | AP | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validación del umbral | 825,358 | 98.2425% | 99.8214% | 0.995016 | 0.997350 | 559 | 10 | 1 |
| Prueba offline final | 1,713,380 | 98.5294% | 99.6283% | 0.994065 | 0.999269 | 1072 | 16 | 4 |

El test offline contiene 1.076 fraudes. Con la política elegida se detectaron 1072 y se omitieron 4, con 16 falsas alarmas. El modelo y la política se congelaron antes de consultar ese bloque. No se utilizó producción.

## Comparación diagnóstica con umbral 0,5

En el test, el umbral 0,5 obtiene F2=0.995743, TP=1076, FP=23 y FN=0. Es ligeramente mejor en F2 que la política elegida en validación, aunque emite más falsas alarmas. **No se cambia el umbral después de ver el test.** Esta diferencia muestra que optimizar en una validación finita no garantiza que la misma regla sea la mejor en otro periodo.

## Evidencia

- Protocolo previo: `docs/EVALUATION_PROTOCOL.md` y `evaluation_protocol.json`.
- Política inmutable por convención y hash: `decision_policy.json`.
- Métricas, curvas y matriz: `summary.json`, `threshold_selection.png`, `offline_evaluation.png`.
- Predicciones por lote y manifiestos en artifacts, excluidos de Git.
- Modelo de decisión joblib y MLflow con umbral incluido; recargados y comprobados en TP, FP, FN y TN reales del test sin publicarlos como ejemplos nuevos.
- Dieciocho pruebas automatizadas aprobadas, incluida comparación de selección F2 con una búsqueda exhaustiva independiente en datos de prueba pequeños.
- Run local `de732b2a5a23477ba0d67dccb89ef67b`, experimento `fraud-offline-evaluation`.

## Alcance

Los datos son simulados y la detección es posterior a transacción porque usa saldos finales. No se ha demostrado prevención en vivo ni generalización a datos bancarios reales. Las métricas de producción simulada se calcularán después, con este mismo modelo y umbral, sin reentrenarlo.

Siguiente etapa: interpretabilidad global y local, seguida de monitoreo. El registro remoto de la decisión final y sus métricas se documentará cuando exista y esté verificado.
