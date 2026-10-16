# Protocolo de optimización

## Comparación congelada antes de evaluar

Se comparan Random Search y TPE sobre XGBoost con histogramas CPU, usando el mismo motor de evaluación. Random Search se implementa con `optuna.samplers.RandomSampler`; no se presenta como una ejecución de `RandomizedSearchCV`. TPE utiliza `TPESampler`, cuatro trials aleatorios iniciales y después sugerencias guiadas. Es el mismo método de búsqueda aleatoria estudiado en clase, con un motor común para controlar la comparación.

Presupuesto: diez configuraciones por método y tres folds por configuración, sesenta ajustes en total. Sin pruning, early stopping ni selección basada en producción. Misma semilla 42, mismo espacio y límites, cuatro hilos y población completa. Los primeros cuatro trials pueden coincidir porque ambos samplers parten de la misma semilla; se ejecutan realmente para ambos, sin reutilizar resultados ni conceder tiempo gratuito a un método.

Métrica objetivo: media no ponderada de Average Precision entre las tres ventanas cronológicas de desarrollo (validación 101–150, 151–200 y 201–250). La desviación entre ventanas se reporta, sin interpretarla como intervalo de confianza. Los steps 251 en adelante no se abren.

## Transformación y desbalance

El tipo se codifica como one-hot con la lista contractual de cinco categorías. Las columnas numéricas se convierten a float32 para reducir memoria. Este encoder no aprende estadísticas y conserva los NaN del ratio de saldo cero: XGBoost utiliza su tratamiento nativo de valores ausentes. El baseline previo usa imputación; esa diferencia se documenta, pero Random Search y TPE reciben exactamente el mismo encoder y los mismos datos entre sí.

Se fija `scale_pos_weight = sqrt(n_negativos / n_positivos)` calculado exclusivamente con las etiquetas del entrenamiento de cada fold. No se calculan pesos con validación ni se balancean sus filas. Es una política fijada, no un hiperparámetro seleccionado mirando resultados. Los scores no se consideran probabilidades calibradas.

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
