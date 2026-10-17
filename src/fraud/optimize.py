"""Equal-budget RandomSampler vs TPE; only steps 1–250, with durable evidence."""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .baseline import metrics
from .boosting import BoostingEncoder, ENCODED_NAMES, training_weight
from .modeling import FeatureContract
from .prepare import ROOT, sha256
from .schema import FEATURES, TARGET, temporal_folds
from .tracking import configure

EXPERIMENT = 'fraud-hyperparameter-search'
SPACE = {
    'n_estimators': {'kind':'int','low':60,'high':180,'step':30},
    'max_depth': {'kind':'int','low':2,'high':6,'step':1},
    'learning_rate': {'kind':'float','low':0.03,'high':0.2,'log':True},
    'min_child_weight': {'kind':'float','low':1.,'high':20.,'log':True},
    'subsample': {'kind':'float','low':0.8,'high':1.0,'log':False},
    'colsample_bytree': {'kind':'float','low':0.8,'high':1.0,'log':False},
    'reg_lambda': {'kind':'float','low':0.1,'high':30.,'log':True},
}


def suggest(trial):
    return {name: (trial.suggest_int(name, spec['low'], spec['high'], step=spec['step'])
                   if spec['kind']=='int' else trial.suggest_float(name, spec['low'], spec['high'], log=spec['log']))
            for name,spec in SPACE.items()}


def fixed_params(threads=4):
    return {'objective':'binary:logistic','tree_method':'hist','max_bin':128,
            'eval_metric':'aucpr','n_jobs':threads,'random_state':42,'verbosity':0}


def prepare_matrix(root, profile):
    """Chunked encoding to disk avoids keeping the full DataFrame in RAM."""
    directory=root/'artifacts/optimization/matrix'
    directory.mkdir(parents=True,exist_ok=True)
    path=root/'data/processed/development.parquet'
    if sha256(path)!=profile['partitions']['development']['parquet_sha256']:
        raise ValueError('Hash de development incorrecto')
    manifest=directory/'manifest.json'
    expected={'data_hash':profile['partitions']['development']['parquet_sha256'],
              'rows':profile['partitions']['development']['rows'],
              'encoded_names':list(ENCODED_NAMES),'dtype':'float32'}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8'))!=expected:
            raise ValueError('La matriz existente tiene otro contrato')
    else:
        n=expected['rows']
        arrays={
            'X':np.lib.format.open_memmap(directory/'X.npy',mode='w+',dtype='float32',shape=(n,len(ENCODED_NAMES))),
            'y':np.lib.format.open_memmap(directory/'y.npy',mode='w+',dtype='int8',shape=(n,)),
            'steps':np.lib.format.open_memmap(directory/'steps.npy',mode='w+',dtype='int16',shape=(n,)),
            'row_ids':np.lib.format.open_memmap(directory/'row_ids.npy',mode='w+',dtype='int64',shape=(n,)),
        }
        offset=0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000):
            data=batch.to_pandas()
            if not data.step.between(1,250).all():raise ValueError('Datos fuera de development')
            end=offset+len(data)
            arrays['X'][offset:end]=BoostingEncoder().transform(data.loc[:,list(FEATURES)])
            for key,col in [('y',TARGET),('steps','step'),('row_ids','row_id')]:
                arrays[key][offset:end]=data[col].to_numpy()
            offset=end
        if offset!=n:raise ValueError('Conteo de matriz incorrecto')
        for a in arrays.values():a.flush()
        del arrays
        manifest.write_text(json.dumps(expected,indent=2),encoding='utf-8')
    return {key:np.load(directory/f'{key}.npy',mmap_mode='r') for key in ['X','y','steps','row_ids']}


def optimize(root:Path=ROOT, trials:int=10, threads:int=4):
    if trials<5 or threads<1:raise ValueError('Usar al menos cinco trials y un hilo')
    reports=root/'artifacts/run_reports/optimization';reports.mkdir(parents=True,exist_ok=True)
    artifacts=root/'artifacts/optimization';artifacts.mkdir(parents=True,exist_ok=True)
    profile=json.loads((root/'artifacts/run_reports/dataset_profile.json').read_text(encoding='utf-8'))
    plan={'trials_per_method':trials,'folds':3,'seed':42,'tpe_startup_trials':4,'tpe_multivariate':False,
          'search_space':SPACE,'fixed_params':fixed_params(threads),'objective':'mean_temporal_average_precision',
          'class_weight_policy':'sqrt(n_negative/n_positive), training fold only',
          'no_early_stopping':True,'no_pruning':True,'sampling':'none',
          'source_sha256':profile['source_sha256'],'validation_steps':[[101,150],[151,200],[201,250]],
          'threshold_partition_used':False,'offline_test_used':False,'production_used':False}
    plan_path=reports/'search_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('El plan congelado no coincide: preservar el estudio antes de cambiarlo')
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if (reports/'summary.json').exists():
        print('Búsquedas ya completadas; se conserva la evidencia.',flush=True)
        return json.loads((reports/'summary.json').read_text(encoding='utf-8'))
    data=prepare_matrix(root,profile)
    folds=list(temporal_folds(pd.Series(data['steps'])))
    mlflow,exp_id,_,_=configure(root,experiment_name=EXPERIMENT,backend='local')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    summaries=[]
    for method in ['random','tpe']:
        sampler_path=artifacts/f'{method}_sampler.joblib'
        sampler=(joblib.load(sampler_path) if sampler_path.exists() else
                 optuna.samplers.RandomSampler(seed=42) if method=='random' else
                 optuna.samplers.TPESampler(seed=42,n_startup_trials=4,multivariate=False))
        study=optuna.create_study(study_name=f'{method}_temporal',direction='maximize',sampler=sampler,
                                  pruner=optuna.pruners.NopPruner(),
                                  storage=f'sqlite:///{(artifacts/"studies.db").as_posix()}',load_if_exists=True)
        if any(t.state==optuna.trial.TrialState.RUNNING for t in study.trials):
            raise RuntimeError('Hay un trial interrumpido; revisar su evidencia antes de continuar')
        if len(study.trials)>trials:raise ValueError('El estudio excede el presupuesto')
        method_dir=reports/method;method_dir.mkdir(exist_ok=True)
        with mlflow.start_run(experiment_id=exp_id,run_name=f'{method}_xgb_temporal_search',
                              tags={'stage':'optimization','method':method,'offline_test_used':'false','production_used':'false'}) as parent:
            parent_id=parent.info.run_id
            mlflow.log_params({'method':method,'trials_budget':trials,'folds':3,'seed':42,**fixed_params(threads)})
            mlflow.log_dict(plan,'search_plan.json')
            def objective(trial):
                params=suggest(trial)
                trial_dir=method_dir/f'trial_{trial.number:02d}';trial_dir.mkdir(exist_ok=True)
                (trial_dir/'params.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
                rows=[];predictions=[];start_trial=perf_counter()
                with mlflow.start_run(experiment_id=exp_id,run_name=f'{method}_trial_{trial.number:02d}',nested=True) as run:
                    mlflow.log_params(params)
                    mlflow.log_param('method',method)
                    for fold_id,(train,valid) in enumerate(folds,start=1):
                        weight=training_weight(data['y'][train])
                        if set(np.unique(data['y'][valid]))!={0,1}:raise ValueError('Validación sin ambas clases')
                        print(f'{method} trial {trial.number+1}/{trials}, fold {fold_id}: {len(train):,} train',flush=True)
                        model=XGBClassifier(**fixed_params(threads),**params,scale_pos_weight=weight)
                        start=perf_counter()
                        model.fit(data['X'][train],data['y'][train])
                        fit_seconds=perf_counter()-start
                        start=perf_counter()
                        score=model.predict_proba(data['X'][valid])[:,1]
                        prediction_seconds=perf_counter()-start
                        result=metrics(data['y'][valid],score)
                        row={'fold':fold_id,'train_rows':len(train),'validation_rows':len(valid),
                             'train_max_step':int(data['steps'][train].max()),
                             'validation_min_step':int(data['steps'][valid].min()),
                             'validation_max_step':int(data['steps'][valid].max()),
                             'scale_pos_weight':weight,'fit_seconds':fit_seconds,'predict_seconds':prediction_seconds,**result}
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(trial_dir/'fold_metrics.csv',index=False)
                        predictions.append(pd.DataFrame({'row_id':data['row_ids'][valid],'step':data['steps'][valid],
                                                         TARGET:data['y'][valid],'fold':fold_id,'score':score}))
                        print(f'  AP={result["average_precision"]:.6f}; fit={fit_seconds:.1f}s',flush=True)
                        mlflow.log_metrics({f'fold_{fold_id}_{k}':v for k,v in row.items()})
                        del model;gc.collect()
                    mean_ap=float(np.mean([r['average_precision'] for r in rows]))
                    std_ap=float(np.std([r['average_precision'] for r in rows],ddof=1))
                    duration=perf_counter()-start_trial
                    record={'method':method,'trial':trial.number,'params':params,'mean_ap':mean_ap,'std_ap':std_ap,
                            'duration_seconds':duration,'run_id':run.info.run_id,'folds':rows}
                    (trial_dir/'result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
                    mlflow.log_metrics({'mean_temporal_average_precision':mean_ap,'std_temporal_average_precision':std_ap,
                                        'duration_seconds':duration})
                    mlflow.log_artifacts(str(trial_dir),artifact_path='trial_evidence')
                    if not study.best_trials or mean_ap>study.best_value:
                        oof=pd.concat(predictions,ignore_index=True)
                        if not oof.row_id.is_unique:raise AssertionError('OOF duplicado')
                        oof.to_parquet(artifacts/f'{method}_best_oof.parquet',index=False)
                    trial.set_user_attr('local_run_id',run.info.run_id)
                    trial.set_user_attr('duration_seconds',duration)
                    trial.set_user_attr('std_ap',std_ap)
                print(f'{method} trial {trial.number+1}: AP media={mean_ap:.6f}',flush=True)
                return mean_ap
            def checkpoint(study,trial):
                joblib.dump(study.sampler,sampler_path)
                study.trials_dataframe().to_csv(method_dir/'trials.csv',index=False)
            remaining=trials-len(study.trials)
            if remaining:study.optimize(objective,n_trials=remaining,n_jobs=1,callbacks=[checkpoint])
            if len(study.trials)!=trials or any(t.state!=optuna.trial.TrialState.COMPLETE for t in study.trials):
                raise RuntimeError('El presupuesto no tiene todos sus trials completos')
            best=study.best_trial
            method_summary={'method':method,'trials':len(study.trials),'best_trial':best.number,
                            'best_ap':best.value,'best_std_ap':best.user_attrs['std_ap'],'best_params':best.params,
                            'objective_seconds':sum(t.user_attrs['duration_seconds'] for t in study.trials),
                            'run_id':parent_id,'trial_run_ids':[t.user_attrs['local_run_id'] for t in study.trials]}
            (method_dir/'summary.json').write_text(json.dumps(method_summary,indent=2),encoding='utf-8')
            mlflow.log_metrics({'best_temporal_ap':best.value,'objective_seconds':method_summary['objective_seconds']})
            mlflow.log_artifact(str(method_dir/'trials.csv'))
            mlflow.log_artifact(str(method_dir/'summary.json'))
            summaries.append(method_summary)
    baseline=json.loads((root/'artifacts/run_reports/baseline/summary.json').read_text(encoding='utf-8'))
    baseline_ap=next(r['average_precision_mean'] for r in baseline['models'] if r['model']=='decision_tree')
    best_method=max(summaries,key=lambda s:s['best_ap'])
    summary={'stage':'optimization','experiment_name':EXPERIMENT,'methods':summaries,'baseline_mean_ap':baseline_ap,
             'best_boosting_method':best_method['method'],'best_boosting_mean_ap':best_method['best_ap'],
             'selected_family':'xgboost' if best_method['best_ap']>baseline_ap+1e-6 else 'decision_tree',
             'selection_rule':'highest mean temporal AP; baseline retained if gain <= 1e-6',
             'source_sha256':profile['source_sha256'],'trials_per_method':trials,'folds':3,
             'threshold_partition_used':False,'offline_test_used':False,'production_used':False,
             'single_seed_comparison':True,'model_role':'candidate_before_threshold_and_offline_test'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for method in ['random','tpe']:
        records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((reports/method).glob('trial_*/result.json'))]
        values=np.array([r['mean_ap'] for r in records]);times=np.cumsum([r['duration_seconds'] for r in records])
        axes[0].plot(np.arange(1,len(values)+1),np.maximum.accumulate(values),marker='o',label=method)
        axes[1].plot(times/60,np.maximum.accumulate(values),marker='o',label=method)
    for ax in axes:
        ax.axhline(baseline_ap,color='gray',linestyle='--',label='Árbol baseline')
        ax.set(ylabel='Mejor AP temporal observada');ax.legend()
    axes[0].set(xlabel='Configuraciones evaluadas',title='Mismo presupuesto de evaluaciones')
    axes[1].set(xlabel='Minutos de evaluación del objetivo',title='Calidad frente a tiempo medido')
    fig.savefig(reports/'search_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
    (reports/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for s in summaries:
        mlflow.MlflowClient().log_artifact(s['run_id'],str(reports/'summary.json'),'comparison')
        mlflow.MlflowClient().log_artifact(s['run_id'],str(reports/'search_comparison.png'),'comparison')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    optimize(trials=args.trials,threads=args.threads)
