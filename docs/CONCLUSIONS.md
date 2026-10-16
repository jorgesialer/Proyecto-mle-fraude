# Conclusiones

1. La comparación temporal acotada favoreció Random Search sobre TPE y el baseline por AP media. Una semilla y diez ensayos por método no permiten generalizar la superioridad del buscador.
2. La política escogida en validación obtiene precisión y recall altos en el test independiente y en el replay completo. El diagnóstico a umbral 0,5 mejora ligeramente F2 offline; conservar la política previa evita seleccionar con el test.
3. La discrepancia contable del origen y el saldo final explican gran parte de las predicciones. Eso exige definir el uso posterior a la transacción y limita la transferencia a fraude real. Explicaciones no equivalen a causalidad ni certifican equidad.
4. El volumen y la prevalencia cambian entre ventanas. La última dispara alertas de distribución sin degradación suficiente para cruzar los límites supervisados. Investigar esa diferencia es más prudente que declarar concept drift o reentrenar automáticamente.
5. ADWIN no detectó cambios en el error observado. Respondió a una alteración deliberada de etiquetas en un escenario separado; esto comprueba sensibilidad, no demuestra drift real ni ausencia del mismo.
6. Próximos pasos fuera del alcance evaluado: validación con datos reales y etiquetas tardías, calibración con otro bloque futuro si se necesitan probabilidades, análisis de costos, seguimiento por segmentos e infraestructura de servicio. Todo nuevo challenger requiere una evaluación temporal nueva.

La aplicación permite consultar localmente los resultados de la simulación y sus reportes.
