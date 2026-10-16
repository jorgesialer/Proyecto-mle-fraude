# Resultado de etapa 3: optimización de hiperparámetros

Se completaron veinte configuraciones de XGBoost (diez por método), tres folds cada una: sesenta ajustes con todos los datos de entrenamiento de cada ventana, sin muestreo. Solo se utilizaron steps 1–250. El protocolo se fijó en `docs/SEARCH_PROTOCOL.md` y `search_plan.json`.

## Comparación medida

| Método | Mejor trial (base 1) | AP media temporal | Desviación entre folds | Minutos de evaluación del objetivo |
| --- | ---: | ---: | ---: | ---: |
| random | 10 | 0.997795 | 0.002000 | 6.97 |
| tpe | 6 | 0.997233 | 0.003047 | 8.99 |

Árbol baseline: AP media temporal 0.994281. Mejor boosting: 0.997795. Diferencia absoluta: 0.003514.

Mejor búsqueda observada: **random**. Familia seleccionada por la regla congelada: **xgboost**. Esto describe el resultado con una semilla y diez configuraciones por método; no demuestra que un método sea superior universalmente. La variación entre folds no es un intervalo de confianza.

## Candidato y decisiones al umbral fijo

Se guardó un pipeline candidato sobre los steps 1–250, recargado con predicciones idénticas. No tiene todavía umbral optimizado ni evaluación offline final.

Predicciones combinadas fuera de muestra de desarrollo: 2,188,776 operaciones. Con umbral fijo 0,5: precision 98.8568%, recall 99.5154%, TP=1643, FP=19, FN=8. Estas decisiones no se usan para ajustar el umbral; los scores ponderados no se declaran calibrados.

## Evidencia

- `random/` y `tpe/`: parámetros, métricas de tres folds y tiempos para cada trial.
- Dos runs principales y veinte runs hijos de búsqueda en MLflow local; un run adicional para el ajuste del candidato.
- Estudios SQLite y estado de samplers en `artifacts/optimization/`, excluidos de Git.
- `search_comparison.png`: mejor AP frente a configuraciones y tiempo medido.
- `candidate.json`: parámetros y trazabilidad del candidato; modelo y flavor MLflow en artifacts.
- Notebook 2 ampliado con protocolo, resultados, comprobaciones y limitaciones.
- Catorce pruebas automatizadas aprobadas antes de entrenar las búsquedas.

## Límites y siguiente paso

Los mejores valores de CV se usan para selección tras probar varias configuraciones, por lo que no son una estimación final independiente. Los bloques de umbral (251–300), prueba offline (301–400) y producción (desde 401) siguen reservados. No se ha declarado un champion productivo.

El tratamiento nativo de NaN y la matriz float32 son iguales entre los métodos de búsqueda; difieren del baseline con imputación y se documentan. La ponderación de clases se calcula solo en cada entrenamiento. El caso sigue siendo posterior a la transacción sobre datos simulados con saldos finales: no acredita prevención en vivo.

Siguiente paso: fijar el candidato, elegir umbral usando exclusivamente su partición y después evaluar una vez el bloque offline. Posteriormente se desarrollarán interpretabilidad y monitoreo.

## Publicación remota pendiente

La revisión automática rechazó publicar el nuevo modelo XGBoost y los veinte trials porque la autorización explícita anterior cubría el baseline. No se ejecutó la publicación de esta etapa. Modelo, métricas, código y ejemplo de ocho filas están preparados localmente para revisión y autorización.
