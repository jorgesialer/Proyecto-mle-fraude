# Plan y evidencias de la rúbrica

| Bloque | Peso | Entregable | Estado inicial |
| --- | --- | --- | --- |
| Repositorio | 10 % | Main, development, PR integrada, release 1.0.0 | Preparación local; remoto y PR pendientes |
| Técnicas avanzadas | 50 % | Baseline, Random Search vs Optuna/TPE, MLflow y SHAP global/local con PDP/ICE | Pendiente de entrenamiento |
| Monitoreo | 30 % | Calidad, drift, desempeño, alertas, app y reporte HTML | Pendiente |
| Documentación | 10 % | README, cuatro notebooks, diccionario y conclusiones | Primer notebook ejecutado y diccionario completos; resto pendiente |

Etapas acordadas: (1) datos e infraestructura, (2) baseline, (3) optimización, (4) interpretabilidad, (5) monitoreo y (6) verificación y entrega. No crear métricas ficticias ni afirmar que un run de preparación es un experimento de entrenamiento.

Se conserva el repositorio anterior de aprobación HMDA. Este proyecto tiene objetivo, modelo, documentación y experimentos propios. El módulo de tracking sigue el patrón reutilizable de `src/mlflow_utils.py` de la V2 del repositorio de jorgesialer/Proyecto-mle-riesgo; no se reutilizan ni el modelo ni el corpus hipotecario.

## Secuencia Git propuesta

`main` contiene hitos revisados. `development` integra trabajo. Cada etapa usa una rama de trabajo y PR hacia development o main según el hito. Antes de entregar: al menos una PR integrada y release v1.0.0 con notas. El README final enlazará los objetos reales cuando existan; este plan no los presenta como completados.
