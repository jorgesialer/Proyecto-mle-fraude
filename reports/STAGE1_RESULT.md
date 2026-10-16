# Resultado de la primera etapa

Preparación local completada el 15 de octubre de 2026. Proyecto independiente: `Proyecto-mle-fraude`. La V2 de aprobación crediticia no se ha modificado.

## Datos preparados

| Partición | Steps observados | Transacciones | Fraudes |
| --- | --- | ---: | ---: |
| Desarrollo | 1–250 | 3.248.292 | 2.841 |
| Validación del umbral | 251–300 | 825.358 | 560 |
| Prueba offline | 301–400 | 1.713.380 | 1.076 |
| Producción | 401–743 | 575.590 | 3.736 |
| Total | 1–743 | 6.362.620 | 8.213 |

Se conservaron todas las filas. Los Parquet están en `data/processed/`, excluidos de Git. El CSV original permanece en su ubicación y está referenciado por el `.env` local, también excluido de Git. Los hashes, perfiles y versión del entorno están en `reports/`.

## Verificaciones realizadas

- Seis pruebas automatizadas aprobaron: fronteras temporales, rechazo de steps inválidos, no influencia del target/alerta/identificador del originador en features, tratamiento del ratio con saldo cero, rechazo de target no binario y folds con horas completas.
- Notebook con 11 celdas de código ejecutadas, sin salidas de error, y exportado a HTML.
- Lectura independiente de todos los Parquet: esquema, pertenencia temporal, conteos, target binario, hashes y unicidad global de los 6.362.620 row_id.
- Cada fold temporal de desarrollo contiene ambas clases y su validación es estrictamente posterior al entrenamiento.
- Figura de prevalencia y volumen revisada visualmente.
- Run MLflow local `006f20c88daf4cb0bc76c561d414c77a`, experimento `fraud-data-preparation`, estado FINISHED. Sus cuatro artefactos fueron consultados y verificados.

## Límites y pendientes

Este run registra preparación de datos; no incluye entrenamiento ni resultados predictivos. El HTML corresponde al primer notebook, no al reporte final de monitoreo.

No se encontraron credenciales DagsHub en las ubicaciones del proyecto revisadas ni en las variables de entorno pertinentes. Por tanto, la conexión remota autenticada no está verificada. No se creó todavía un repositorio remoto nuevo de GitHub ni una PR. No se afirma que estén disponibles por existir enlaces del proyecto anterior.

Siguiente etapa: entrenar y evaluar un baseline usando las particiones y el contrato congelados, registrar sus resultados y preparar la comparación Random Search/Optuna. Producción seguirá excluida de la selección inicial.
