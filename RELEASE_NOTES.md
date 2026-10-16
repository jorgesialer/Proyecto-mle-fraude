# Versiones del proyecto

## 1.0.2 — Documentación técnica para portafolio

El README, las guías y los notebooks describen el comportamiento del proyecto con frases activas y sin primera persona. Las instrucciones de instalación, configuración y uso emplean el modo imperativo. Los reportes HTML reflejan la misma revisión editorial. El código, las salidas de los notebooks y el modelo permanecen sin cambios.

## 1.0.1 — Presentación y evidencias

La versión simplifica la estructura de la entrega y reúne las dependencias en `requirements.txt`. El entorno local y MLflow conservan los reportes de avance y los detalles por intento. La documentación utiliza una redacción más directa y la referencia del dataset apunta a Kaggle.

La documentación incorpora cinco capturas reales del panel y de MLflow, además de los gráficos de evaluación y SHAP. Los cuatro notebooks conservan su código y sus resultados. No se volvió a entrenar el modelo ni se cambiaron las métricas.

## 1.0.0 — Primera entrega

Detección de fraude posterior a transacción sobre PaySim, con evaluación temporal independiente y monitoreo histórico completo.

- Cuatro notebooks ejecutados: preprocesamiento, entrenamiento/optimización/evaluación, interpretabilidad y monitoreo.
- Baseline y comparación de Random Search/TPE: veinte configuraciones y sesenta ajustes temporales de XGBoost.
- Modelo y umbral congelados antes del test y producción. Offline: TP 1.072, FN 4, FP 16. Replay: TP 3.715, FN 21, FP 7.
- TreeSHAP global/local, permutación y PDP/ICE con supuestos y limitaciones explícitos.
- App Streamlit, reportes HTML y Evidently, PSI/KS con BH, ADWIN, política de alertas y prueba sintética separada.
- MLflow/DagsHub: experimentos verificados y modelo fraud-paysim-posttransaction v1, READY, Production para replay educativo. Sin servicio en vivo.
- Veintidós pruebas aprobadas, cuatro notebooks sin errores, filtro de app comprobado, paridad del modelo remoto y hashes verificados.

Consulta el README para reproducir el proyecto y localizar los enlaces MLflow, informes y diccionario. Para consultar el panel: `python -m streamlit run app.py`. GitHub muestra HTML como código; descarga los archivos y ábrelos localmente. Los ZIP/TAR de código de la release contienen los reportes, sin CSV completo, credenciales ni binarios del modelo.

El modelo binario se recupera desde el run MLflow enlazado en README. Requiere configuración local de autenticación si DagsHub la solicita. La V2 de riesgo crediticio se conserva intacta. Esta entrega no certifica desempeño en fraude real ni equidad demográfica.
