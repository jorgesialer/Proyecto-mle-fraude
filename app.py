"""Read-only dashboard of the historical replay. Run with Streamlit."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
REPORTS=ROOT/'reports'
st.set_page_config(page_title='Fraude · monitoreo',page_icon='📊',layout='wide')
st.title('Detección de fraude · monitoreo')
st.caption('PaySim · simulación histórica · modelo y umbral congelados · etiquetas históricas disponibles')
path=REPORTS/'resultados.json'
if not path.exists():
    st.info('Genera primero las evidencias: python -m src.fraud.monitor')
    st.stop()
results=json.loads(path.read_text(encoding='utf-8'))
summary=results['monitoring']
windows=pd.DataFrame(results['windows'])
drift=pd.DataFrame(results['drift'])
quality=pd.DataFrame(results['quality'])
labels=['Todo el periodo']+[f'Steps {r.step_min}–{r.step_max}' for r in windows.itertuples()]
choice=st.sidebar.selectbox('Periodo',labels)
selected=None if choice==labels[0] else windows.iloc[labels.index(choice)-1]
row=summary['metrics'] if selected is None else selected
rows=summary['rows'] if selected is None else int(selected['rows'])
cols=st.columns(4)
cols[0].metric('Transacciones',f'{rows:,}')
cols[1].metric('Precisión',f'{row["precision"]:.2%}')
cols[2].metric('Recall',f'{row["recall"]:.2%}')
cols[3].metric('F2',f'{row["f2"]:.4f}')
st.sidebar.write(f'Umbral fijo: {summary["threshold"]:.6f}')
st.sidebar.caption(f'Modelo SHA256: {summary["model_sha256"][:16]}…')
performance,distribution,explanation,evidence=st.tabs(['Desempeño','Calidad y drift','Explicaciones','Reportes y método'])
with performance:
    st.write(f'Fraudes detectados: **{int(row["tp"]):,}** · omitidos: **{int(row["fn"]):,}** · falsas alarmas: **{int(row["fp"]):,}**')
    st.line_chart(windows.set_index('step_max')[['precision','recall','f2']])
    st.dataframe(windows,hide_index=True)
    st.caption('Las ventanas tienen distintos volúmenes. El replay no reentrena ni cambia el umbral.')
with distribution:
    shown=drift if selected is None else drift[drift.window==selected['window']]
    st.write(f'**{int(shown.drift_alert.sum())}** combinaciones variable/ventana con alerta según la política propia.')
    st.image(str(ROOT/'evidencias/drift_heatmap.png'))
    st.dataframe(shown,hide_index=True)
    st.dataframe(quality if selected is None else quality[quality.window==selected['window']],hide_index=True)
    st.caption('PSI ≥ 0,2 o KS con efecto ≥ 0,1 y corrección BH al 5 %. Referencia fija: 50.000 filas uniformes de steps 301–400. NaN del ratio es estructural.')
with explanation:
    st.image(str(ROOT/'evidencias/shap_global.png'))
    st.image(str(ROOT/'evidencias/shap_local_cases.png'))
    st.caption('Explicaciones del test offline. TreeSHAP suma en log-odds; los scores no están calibrados. Asociación no implica causalidad.')
with evidence:
    signals=results['adwin']
    st.write(f'ADWIN sobre errores observados: **{len(signals["observed_error_changes"])}** señales.')
    st.write('Una señal requiere investigación; no prueba por sí sola concept drift. Ausencia de señal tampoco demuestra estabilidad.')
    with st.expander('Prueba controlada de sensibilidad — escenario sintético separado'):
        st.json(signals['controlled_scenario'])
        st.caption('Se invirtió deliberadamente el 15 % de etiquetas después de la mitad del replay. No forma parte de las métricas productivas.')
    for filename,label in [('04_monitoreo.html','Notebook de monitoreo ejecutado'),('evidently_drift.html','Reporte Evidently de drift')]:
        report=REPORTS/filename
        if report.exists():st.download_button(label,report.read_bytes(),file_name=filename,mime='text/html')
    st.download_button('Métricas por ventana (CSV)',windows.to_csv(index=False),file_name='windows.csv',mime='text/csv')
    st.write('Ante una alerta: comprobar calidad y etiquetas, revisar errores por tipo/monto y confirmar persistencia. Evaluar un challenger con datos posteriores etiquetados y validación temporal antes de promoverlo con revisión humana.')
    st.caption('Aplicación local de análisis histórico; no está conectada a transacciones reales.')
