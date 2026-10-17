"""Build a compact delivery from detailed, local execution records."""
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def export_delivery(root:Path=ROOT):
    internal=root/'artifacts/run_reports'
    output=root/'reports';output.mkdir(parents=True,exist_ok=True)
    def read(name):return json.loads((internal/name).read_text(encoding='utf-8'))
    def rows(name):
        frame=pd.read_csv(internal/name)
        return frame.astype(object).where(pd.notna(frame),None).to_dict(orient='records')
    bundle={
        'baseline':read('baseline/summary.json'),
        'optimization':read('optimization/summary.json'),
        'candidate':read('optimization/candidate.json'),
        'evaluation':read('evaluation/summary.json'),
        'interpretability':read('interpretability/summary.json'),
        'monitoring':read('monitoring/summary.json'),
        'windows':rows('monitoring/windows.csv'),
        'drift':rows('monitoring/feature_drift.csv'),
        'quality':rows('monitoring/quality.csv'),
        'adwin':read('monitoring/adwin.json'),
    }
    for name in ['baseline/remote_tracking.json','optimization/remote_tracking.json','final_remote_tracking.json']:
        if (internal/name).exists():bundle.setdefault('tracking',{})[name]=read(name)
    if sum(w['rows'] for w in bundle['windows'])!=bundle['monitoring']['rows']:
        raise ValueError('Las ventanas no coinciden con el periodo de monitoreo')
    for name in ['tp','fp','fn','tn']:
        if sum(w[name] for w in bundle['windows'])!=bundle['monitoring']['metrics'][name]:
            raise ValueError('Las métricas por ventana no coinciden con el resumen')
    model_hash=bundle['evaluation']['decision_model_sha256']
    if any(bundle[s]['model_sha256']!=model_hash for s in ['interpretability','monitoring']):
        raise ValueError('Los resultados corresponden a modelos diferentes')
    (output/'resultados.json').write_text(json.dumps(bundle,indent=2,allow_nan=False),encoding='utf-8')
    figures=root/'evidencias';figures.mkdir(exist_ok=True)
    for name in ['evaluation/offline_evaluation.png',
        'interpretability/shap_global.png','interpretability/shap_local_cases.png',
        'monitoring/drift_heatmap.png','optimization/search_comparison.png']:
        path=internal/name
        if path.exists():shutil.copy2(path,figures/path.name)
    html=internal/'monitoring/evidently_drift.html'
    if html.exists():shutil.copy2(html,output/'evidently_drift.html')
    return bundle


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--html',action='store_true',help='Export the recorded notebook outputs as HTML')
    args=parser.parse_args()
    bundle=export_delivery()
    if args.html:
        import nbformat
        from nbconvert import HTMLExporter
        for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
            html,_=HTMLExporter().from_notebook_node(nbformat.read(path,as_version=4))
            (ROOT/'reports'/f'{path.stem}.html').write_text(html,encoding='utf-8')
    print(json.dumps({'rows':bundle['monitoring']['rows'],'report':'reports/resultados.json'}))
