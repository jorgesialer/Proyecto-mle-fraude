# Ficha del modelo

## Objetivo y entrenamiento

El modelo identifica transacciones fraudulentas en PaySim después de que la operación fue registrada. Usa saldos finales, por lo que sus resultados corresponden a ese momento del análisis.

El proceso de selección compara Random Search y TPE con diez configuraciones por método y tres ventanas temporales. La configuración final utiliza XGBoost con 90 árboles y profundidad 3. El modelo entrena en steps 1–250; la validación fija el umbral en 251–300; el test offline evalúa en 301–400 y el monitoreo analiza los datos desde 401. Los bloques reservados no participan en el entrenamiento.

El modelo recibe trece predictores preparados, descritos en el [diccionario de datos](README.md#diccionario-de-datos). El pipeline excluye la etiqueta, la regla `isFlaggedFraud`, los identificadores y el step. El umbral elegido fue 0,9741922616958618; los scores no están calibrados como probabilidades.

## Resultados

| Periodo | Precisión | Recall | F2 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test offline | 98,53 % | 99,63 % | 0,994065 | 1.072 | 16 | 4 |
| Producción simulada | 99,81 % | 99,44 % | 0,995125 | 3.715 | 7 | 21 |

El umbral 0,5 obtiene F2 ligeramente mayor en el test offline. El modelo conserva el umbral seleccionado en validación para evitar ajustes con los resultados del test.

## Interpretación y limitaciones

SHAP señala como variables principales la discrepancia del débito y el saldo final de origen. Las explicaciones globales usan 20.000 transacciones de prueba, con 16 fraudes. El notebook 03 incluye permutación, PDP/ICE y casos de aciertos y errores.

PaySim contiene relaciones contables que ayudan a distinguir los fraudes. Un buen resultado aquí no garantiza el mismo comportamiento con datos reales. La correlación entre variables también limita la interpretación de permutaciones y curvas. El dataset no contiene los atributos necesarios para evaluar equidad demográfica.

Una ventana presentó cambios de distribución sin cruzar los límites de desempeño definidos. ADWIN no emitió señales sobre los errores observados. La alteración deliberada de etiquetas se hizo en un escenario aparte para comprobar la respuesta del detector.

## Registro del modelo

[Modelo y evaluación en DagsHub](https://dagshub.com/jorgesialer/Proyecto-mle-riesgo.mlflow/#/experiments/3/runs/5cc382370d9f41eda4e300af13eb1ba1).

- Nombre: `fraud-paysim-posttransaction`.
- Versión 1, estado READY, etapa Production para producción simulada.
- URI: `runs:/5cc382370d9f41eda4e300af13eb1ba1/model`.
- Resultados de evaluación y monitoreo: [resultados.json](reports/resultados.json).

La aplicación se ejecuta localmente.

Consulta el [README](README.md) para instalar la aplicación y reproducir el análisis.
