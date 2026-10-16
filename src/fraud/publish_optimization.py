"""Publish completed search evidence and its candidate, without exporting full data."""
import json
from pathlib import Path
import time
import os

from mlflow.entities import Metric, Param

from .optimize import EXPERIMENT
from .prepare import ROOT
from .tracking import configure


def publish(root:Path=ROOT):
    reports=root/'reports/optimization'
    summary=json.loads((reports/'summary.json').read_text(encoding='utf-8'))
    candidate=json.loads((reports/'candidate.json').read_text(encoding='utf-8'))
    mlflow,exp_id,_,_=configure(root,experiment_name=EXPERIMENT,backend='dagshub')
    client=mlflow.MlflowClient()
    state_path=reports/'remote_tracking.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {
        'experiment_name':EXPERIMENT,'experiment_id':exp_id,'methods':{},'trial_runs':{},'remote_write_verified':False}
    def save():state_path.write_text(json.dumps(state,indent=2),encoding='utf-8')
    def batch(run_id,params,metrics):
        timestamp=int(time.time()*1000)
        client.log_batch(run_id,params=[Param(str(k),str(v)) for k,v in params.items()],
                         metrics=[Metric(str(k),float(v),timestamp,0) for k,v in metrics.items()])
    for method in summary['methods']:
        name=method['method']
        if name not in state['methods']:
            run=client.create_run(exp_id,run_name=f'{name}_xgb_temporal_search',
                tags={'stage':'optimization','method':name,'source_local_run_id':method['run_id'],
                      'offline_test_used':'false','production_used':'false'})
            state['methods'][name]={'run_id':run.info.run_id};save()
        parent_id=state['methods'][name]['run_id']
        batch(parent_id,{'method':name,'trials_budget':summary['trials_per_method'],'folds':3,'seed':42},
              {'best_temporal_ap':method['best_ap'],'best_std_ap':method['best_std_ap'],
               'objective_seconds':method['objective_seconds']})
        for path in sorted((reports/name).glob('trial_*/result.json')):
            record=json.loads(path.read_text(encoding='utf-8'))
            key=f"{name}_{record['trial']}"
            if key not in state['trial_runs']:
                run=client.create_run(exp_id,run_name=f"{name}_trial_{record['trial']:02d}",
                    tags={'mlflow.parentRunId':parent_id,'stage':'optimization_trial','method':name,
                          'source_local_run_id':record['run_id']})
                state['trial_runs'][key]={'run_id':run.info.run_id,'verified':False};save()
            child=state['trial_runs'][key]
            if not child['verified']:
                values={'mean_temporal_average_precision':record['mean_ap'],'std_temporal_average_precision':record['std_ap'],
                        'duration_seconds':record['duration_seconds']}
                for fold in record['folds']:
                    values.update({f"fold_{fold['fold']}_{k}":v for k,v in fold.items()})
                batch(child['run_id'],{'method':name,**record['params']},values)
                client.log_artifact(child['run_id'],str(path),'trial_evidence')
                client.set_terminated(child['run_id'])
                remote=client.get_run(child['run_id'])
                if abs(remote.data.metrics['mean_temporal_average_precision']-record['mean_ap'])>1e-12:
                    raise RuntimeError('La métrica remota del trial no coincide')
                child['verified']=True;save()
            print(f"Trial remoto verificado: {key}",flush=True)
        for filename in ['search_plan.json','summary.json','search_comparison.png','STAGE3_RESULT.md']:
            p=reports/filename
            if p.exists():client.log_artifact(parent_id,str(p),'comparison')
        client.set_terminated(parent_id)
    if 'candidate' not in state:
        run=client.create_run(exp_id,run_name='candidate_development_refit',
            tags={'stage':'candidate_refit','role':'not_final_champion','source_local_run_id':candidate['run_id']})
        state['candidate']={'run_id':run.info.run_id};save()
    candidate_id=state['candidate']['run_id']
    if not state['candidate'].get('verified'):
        batch(candidate_id,{'family':candidate['family'],'selection_method':candidate['selection_method'],
                           'training_step_max':250,**candidate['params']},{'refit_seconds':candidate['refit_seconds']})
        client.log_artifacts(candidate_id,str(root/'artifacts/optimization/candidate_mlflow_model'),'model')
        client.log_artifact(candidate_id,str(reports/'candidate.json'),'candidate_evidence')
        client.set_terminated(candidate_id)
        paths={p.path for p in client.list_artifacts(candidate_id,'model')}
        if not {'model/MLmodel','model/model.pkl'}.issubset(paths):raise RuntimeError('Modelo remoto incompleto')
        state['candidate']['verified']=True
    uri=os.environ['MLFLOW_TRACKING_URI'].rstrip('/')
    for entry in [*state['methods'].values(),state['candidate']]:
        entry['run_url']=f"{uri}/#/experiments/{exp_id}/runs/{entry['run_id']}"
        if client.get_run(entry['run_id']).info.status!='FINISHED':raise RuntimeError('Run remoto no finalizado')
    state['remote_write_verified']=True
    state['verified_trial_count']=len(state['trial_runs'])
    state['candidate']['model_role']='candidate_before_threshold_and_offline_test'
    save()
    print(json.dumps(state,indent=2),flush=True)
    return state


if __name__=='__main__':
    try:publish()
    except Exception as exc:
        print(json.dumps({'remote_write_verified':False,'error_type':type(exc).__name__}))
        raise SystemExit(1)
