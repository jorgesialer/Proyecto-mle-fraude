# Diccionario y contrato de variables

Caso de uso: detección de fraude después de registrar la transacción. Los saldos finales son admisibles bajo este contrato. Este proyecto no afirma prevenir una operación antes de que ocurra.

## Fuente y columnas originales

Dataset PaySim entregado por el profesor, archivo `PS_20174392719_1491204439457_log.csv`.

| Columna | Significado | Uso |
| --- | --- | --- |
| `step` | Hora de la simulación | Separación temporal y monitoreo; excluida de predictores |
| `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | Predictor categórico |
| `amount` | Monto de la transacción en moneda local simulada | Predictor numérico |
| `nameOrig` | Identificador del originador | Excluido para evitar memorización |
| `oldbalanceOrg` | Saldo inicial del originador; se conserva la grafía del CSV | Predictor numérico |
| `newbalanceOrig` | Saldo final del originador | Predictor posterior a transacción |
| `nameDest` | Identificador del destinatario | Solo se extrae el prefijo M de merchant |
| `oldbalanceDest` | Saldo inicial del destinatario | Predictor; ceros de merchant no significan saldo conocido |
| `newbalanceDest` | Saldo final del destinatario | Predictor posterior a transacción; misma limitación |
| `isFraud` | 1 si la transacción está etiquetada como fraude | Target; nunca predictor |
| `isFlaggedFraud` | Alerta de la regla existente del simulador | Excluida de predictores; futura referencia comparativa |

## Variables derivadas

| Variable | Definición | Tratamiento |
| --- | --- | --- |
| `origin_balance_change` | oldbalanceOrg − newbalanceOrig | Cambio de saldo, sin interpretación causal |
| `destination_balance_change` | newbalanceDest − oldbalanceDest | Cambio observado del destinatario |
| `amount_to_origin_balance` | amount / oldbalanceOrg | NaN si el denominador es cero; no usar infinito |
| `origin_zero_balance` | oldbalanceOrg == 0 | Indicador explícito |
| `destination_is_merchant` | nameDest empieza con M | Distingue saldos no informados del destinatario |
| `origin_debit_discrepancy` | origin_balance_change − amount | Interpretar por tipo: CASH_IN aumenta el saldo |
| `destination_credit_discrepancy` | destination_balance_change − amount | No asumir que un merchant informa sus saldos |

Los NaN generados por el ratio son intencionales. La imputación se ajustará dentro del pipeline de entrenamiento de cada fold. No se ajustan escaladores ni codificadores sobre todo el dataset.

`row_id` conserva la posición original, de base cero, para trazabilidad y pruebas de unicidad. No es predictor. En cada Parquet se guardan features, `row_id`, `step` y target; estos tres últimos se separarán de X antes de entrenar.

## Particiones congeladas

| Partición | Steps | Propósito |
| --- | --- | --- |
| development | 1–250 | Baseline, comparación de modelos y búsqueda |
| threshold_validation | 251–300 | Elección de umbral con modelo ya seleccionado |
| offline_test | 301–400 | Evaluación final sin ajuste |
| production | 401–744, hasta el último step observado | Replay de monitoreo; no selección inicial |

El corte 400/401 procede de la consigna. Los cortes internos son decisiones metodológicas. La validación interna usa entrenamiento creciente y ventanas 101–150, 151–200 y 201–250. Cada hora queda completa en un lado del corte. Antes de entrenar debe verificarse que cada fold contenga ambas clases.

El perfilado incluye producción únicamente para controles de integridad y descripción del archivo. No debe convertirse en ajuste de hiperparámetros ni umbral.
