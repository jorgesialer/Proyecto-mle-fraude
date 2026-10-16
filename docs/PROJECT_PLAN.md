# Estructura de la entrega y uso de Git

La consigna pide cuatro notebooks, datos, aplicación de monitoreo, reporte HTML, documentación y experimentos en MLflow. No exige un archivo por cada resultado intermedio ni cuatro archivos de dependencias.

| Parte | Peso | Dónde revisarla |
| --- | --- | --- |
| Repositorio | 10 % | main, development, PR integrada y release 1.0.0 |
| Optimización e interpretación | 50 % | Notebooks 02 y 03, experimentos MLflow y gráficos |
| Monitoreo | 30 % | Notebook 04, app.py, HTML y capturas |
| Documentación | 10 % | README, diccionario, métodos y conclusiones |

## Estrategia de Git

Usé `development` para integrar la preparación, el baseline, la optimización y el monitoreo. Esos cambios se pasaron a `main` en la [PR #1](https://github.com/jorgesialer/Proyecto-mle-fraude/pull/1). La release 1.0.0 conserva la primera entrega. Los cambios posteriores de documentación y presentación se registran en otra PR para que sea fácil revisar qué cambió.

## Qué archivos son de apoyo

`src` reúne funciones compartidas por los notebooks. `tests` comprueba el código. Los JSON y CSV de `reports` permiten construir las tablas y abrir la aplicación sin entrenar de nuevo. Las capturas están en `docs/evidencias`.

Los datos procesados, los modelos y la base MLflow local se generan al ejecutar el proyecto y no se suben a GitHub. Los detalles por intento y reportes de avance anteriores se conservaron localmente y en MLflow; la entrega actual muestra los resúmenes y resultados finales.
