"""Publish the frozen decision model and completed reports; never publish raw data."""
import json
import os
from pathlib import Path
import time
from mlflow.entities import Metric,Param
from .prepare import ROOT,sha256
from .tracking import configure


def publish(root:Path=ROOT):
    evaluation=json.loads((root/'reports/evaluation/summary.json').read_text(encoding='utf-8'))
    xai=json.loads((root/'reports/interpretability/summary.json').read_text(encoding='utf-8'))
    monitoring=json.loads((root/'reports/monitoring/summary.json').read_text(encoding='utf-8'))
    model_path=root/'artifacts/evaluation/decision_model.joblib'
    if not evaluation['decision_model_sha256']==xai['model_sha256']==monitoring['model_sha256']==sha256(model_path):
        raise ValueError('Evidencias de modelos diferentes')
    mlflow,exp_id,_,_=configure(root,experiment_name='fraud-final-delivery',backend='dagshub')
    client=mlflow.MlflowClient();path=root/'reports/final_remote_tracking.json'
    state=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'experiment_id':exp_id,'runs':{},'remote_write_verified':False}
    def save():path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    for stage,summary,metric_values in [('evaluation',evaluation,evaluation['offline_metrics']),
        ('interpretability',xai,{'additivity_max_abs_error':xai['additivity_max_abs_error']}),
        ('monitoring',monitoring,{**monitoring['metrics'],'rows':monitoring['rows'],'observed_adwin_signals':monitoring['observed_adwin_signals']})]:
        if stage not in state['runs']:
            run=client.create_run(exp_id,run_name=f'final_{stage}',tags={'stage':stage,
                'prediction_moment':'post_transaction','source_local_run_id':summary['run_id'],
                'deployment':'historical_simulation','model_sha256':evaluation['decision_model_sha256']})
            state['runs'][stage]={'run_id':run.info.run_id,'verified':False};save()
        entry=state['runs'][stage];run_id=entry['run_id']
        if not entry['verified']:
            params={'model_sha256':evaluation['decision_model_sha256'],'threshold':evaluation['threshold']}
            client.log_batch(run_id,params=[Param(k,str(v)) for k,v in params.items()],
                metrics=[Metric(k,float(v),int(time.time()*1000),0) for k,v in metric_values.items()])
            folder='interpretability' if stage=='interpretability' else stage
            client.log_artifacts(run_id,str(root/f'reports/{folder}'),'evidence')
            if stage=='evaluation':
                client.log_artifacts(run_id,str(root/'artifacts/evaluation/mlflow_model'),'model')
                client.log_artifact(run_id,str(model_path),'portable_model')
                client.log_artifact(run_id,str(root/'model-card.md'),'documentation')
            client.set_terminated(run_id)
            remote=client.get_run(run_id)
            if remote.info.status!='FINISHED' or any(abs(remote.data.metrics[k]-float(v))>1e-10 for k,v in metric_values.items()):
                raise RuntimeError('La evidencia remota no coincide')
            entry['verified']=True;save()
        entry['run_url']=f"{os.environ['MLFLOW_TRACKING_URI'].rstrip('/')}/#/experiments/{exp_id}/runs/{run_id}"
        print(f'{stage}: run remoto verificado',flush=True)
    run_id=state['runs']['evaluation']['run_id']
    destination=root/'artifacts/remote_final_verification';destination.mkdir(parents=True,exist_ok=True)
    downloaded=Path(client.download_artifacts(run_id,'portable_model/decision_model.joblib',str(destination)))
    if sha256(downloaded)!=evaluation['decision_model_sha256']:raise RuntimeError('Hash remoto diferente')
    pkl=Path(client.download_artifacts(run_id,'model/model.pkl',str(destination)))
    if sha256(pkl)!=sha256(root/'artifacts/evaluation/mlflow_model/model.pkl'):raise RuntimeError('Hash MLflow diferente')
    state.update({'model_uri':f'runs:/{run_id}/model','model_sha256':evaluation['decision_model_sha256'],
        'downloaded_model_hash_verified':True,'remote_write_verified':True})
    # Registry support depends on the tracking service. Never claim registration on failure.
    name='fraud-paysim-posttransaction'
    if not state.get('registry',{}).get('verified'):
        try:
            existing=client.search_registered_models(filter_string=f"name = '{name}'")
            if not existing:client.create_registered_model(name,description='Modelo educativo PaySim, evaluación temporal y producción histórica simulada.')
            matches=client.search_model_versions(f"name = '{name}'")
            version=next((v for v in matches if v.run_id==run_id),None)
            if version is None:version=client.create_model_version(name,source=state['model_uri'],run_id=run_id)
            checked=client.get_model_version(name,version.version)
            if checked.run_id!=run_id:raise RuntimeError('Registro de otro modelo')
            state['registry']={'verified':True,'name':name,'version':str(version.version),'live_deployment':False}
        except Exception as exc:
            state['registry']={'verified':False,'error_type':type(exc).__name__,'model_available_at_verified_run_uri':True}
    save();print(json.dumps(state,indent=2));return state


if __name__=='__main__':
    try:publish()
    except Exception as exc:
        print(json.dumps({'remote_write_verified':False,'error_type':type(exc).__name__}))
        raise SystemExit(1)
