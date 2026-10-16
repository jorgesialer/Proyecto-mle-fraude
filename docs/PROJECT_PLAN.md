# Rúbrica y evidencia

| Bloque | Peso | Evidencia y estado |
| --- | --- | --- |
| Repositorio | 10 % | [Repo independiente](https://github.com/jorgesialer/Proyecto-mle-fraude), main/development publicados, [PR #1 integrada](https://github.com/jorgesialer/Proyecto-mle-fraude/pull/1); entrega 1.0.0 |
| Optimización e interpretación | 50 % | Baseline, Random Search/TPE, 20 trials/60 folds, modelo congelado, MLflow, SHAP global/local, permutación, PDP/ICE; completo |
| Monitoreo | 30 % | Replay 401–743, calidad, PSI/KS, ADWIN, política, app Streamlit, HTML Evidently y notebook; completo localmente |
| Documentación | 10 % | README, diccionario, protocolos, ficha del modelo, conclusiones y cuatro notebooks ejecutados |

Se conserva V2. La conexión MLflow se reutiliza con experimentos independientes. El proyecto nuevo corresponde a release 1.0.0 según la consigna; la publicación se acredita con enlaces reales. No se inventan PR, releases, despliegues ni desempeño real.

La evidencia del umbral se guarda antes de abrir test; las alertas se fijan antes de producción. Los hashes enlazan datos, modelo, política, predicciones y resultados. La selección terminó antes de interpretabilidad/monitoreo. El origen y alcance de las muestras se declaran en cada notebook.
