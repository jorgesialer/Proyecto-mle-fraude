# Protocolo de optimización

## Comparación congelada antes de evaluar

El módulo de optimización compara Random Search y TPE sobre XGBoost con histogramas CPU y un motor de evaluación común. Random Search utiliza `optuna.samplers.RandomSampler`. TPE utiliza `TPESampler`, comienza con cuatro trials aleatorios y continúa con sugerencias guiadas por los resultados anteriores.

Presupuesto: diez configuraciones por método y tres folds por configuración, sesenta ajustes en total. Sin pruning, early stopping ni selección basada en producción. Misma semilla 42, mismo espacio y límites, cuatro hilos y población completa. Los primeros cuatro trials pueden coincidir porque ambos samplers parten de la misma semilla; se ejecutan realmente para ambos, sin reutilizar resultados ni conceder tiempo gratuito a un método.

Métrica objetivo: media no ponderada de Average Precision entre las tres ventanas cronológicas de desarrollo (validación 101–150, 151–200 y 201–250). La desviación entre ventanas se reporta, sin interpretarla como intervalo de confianza. Los steps 251 en adelante no se abren.

## Transformación y desbalance

El encoder representa el tipo mediante one-hot con las cinco categorías del contrato y convierte las columnas numéricas a float32 para reducir memoria. La transformación conserva los NaN del ratio de saldo cero y no aprende estadísticas: XGBoost aplica su tratamiento nativo de valores ausentes. El baseline utiliza imputación. Random Search y TPE comparten exactamente el mismo encoder y los mismos datos.

Cada fold calcula `scale_pos_weight = sqrt(n_negativos / n_positivos)` exclusivamente con sus etiquetas de entrenamiento. La validación conserva todas sus filas y no participa en el cálculo de pesos. Esta política permanece fija durante la comparación. Los scores no constituyen probabilidades calibradas.

## Espacio de búsqueda

| Parámetro | Dominio |
| --- | --- |
| n_estimators | 60, 90, 120, 150, 180 |
| max_depth | Enteros 2–6 |
| learning_rate | 0,03–0,2, log-uniforme |
| min_child_weight | 1–20, log-uniforme |
| subsample | 0,8–1, uniforme |
| colsample_bytree | 0,8–1, uniforme |
| reg_lambda | 0,1–30, log-uniforme |

Fijos: objective=binary:logistic, tree_method=hist, max_bin=128 y random_state=42. Se miden duración por fold y por trial; mismo número de evaluaciones no implica idéntico tiempo. Los gráficos comparan calidad frente a configuraciones y tiempo medido del objetivo.

## Evidencia y límites

Cada trial guarda parámetros, cortes, pesos, métricas por fold, duración y run MLflow local. Los estudios SQLite y el estado del sampler se conservan en artifacts para evitar repetir búsquedas completadas. No se registra una mejora hasta terminar la comparación.

Se selecciona por AP media temporal entre el mejor XGBoost y el árbol baseline; si la mejora no supera 1e-6 se conserva el baseline. Las mejores métricas de CV tras buscar múltiples configuraciones son optimistas para estimar generalización: la evaluación final seguirá en el bloque offline reservado, después de elegir el umbral en su propio bloque.

Esta es una comparación acotada con una semilla. No demuestra superioridad universal de un optimizador, ni que se haya encontrado un óptimo global. La simulación y el uso de saldos finales limitan la interpretación de resultados altos.

Referencias: notebooks de las sesiones 3–4; [Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html); [API XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier).
