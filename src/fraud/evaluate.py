"""Freeze a validation-only threshold, then evaluate the independent offline block."""
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import precision_recall_curve, ConfusionMatrixDisplay, confusion_matrix

from .baseline import metrics
from .decision import choose_f2_threshold, FrozenThresholdClassifier
from .prepare import ROOT, sha256
from .schema import FEATURES, TARGET
from .tracking import configure

EXPERIMENT='fraud-offline-evaluation'
PROTOCOL={'criterion':'max_f2','beta':2,'tie_break':'highest_threshold','comparison':'score >= threshold',
          'training_steps':[1,250],'threshold_steps':[251,300],'offline_steps':[301,400],
          'production_used':False,'retrain_after_threshold':False}


def score_partition(root,name,pipeline,profile,artifacts):
    """Batch inference, with a cache tied to exact model and partition hashes."""
    if name not in ['threshold_validation','offline_test']:
        raise ValueError('La evaluación offline no admite producción')
    source=root/f'data/processed/{name}.parquet'
    partition_hash=sha256(source)
    if partition_hash!=profile['partitions'][name]['parquet_sha256']:raise ValueError('Hash de partición incorrecto')
    cache=artifacts/f'{name}_predictions.parquet';manifest=artifacts/f'{name}_scoring.json'
    candidate_hash=sha256(root/'artifacts/optimization/pipeline_candidate.joblib')
    expected={'partition_sha256':partition_hash,'candidate_sha256':candidate_hash}
    if cache.exists() or manifest.exists():
        if not cache.exists() or not manifest.exists():raise RuntimeError('Caché de evaluación incompleta')
        metadata=json.loads(manifest.read_text(encoding='utf-8'))
        if any(metadata[k]!=v for k,v in expected.items()) or sha256(cache)!=metadata['predictions_sha256']:
            raise ValueError('La evidencia previa no corresponde a estos datos/modelo')
        return pd.read_parquet(cache),metadata
    if name=='offline_test' and not (root/'artifacts/run_reports/evaluation/decision_policy.json').is_file():
        raise RuntimeError('No abrir test offline antes de congelar la política')
    rows=[];seconds=0.
    for batch in pq.ParquetFile(source).iter_batches(batch_size=100_000):
        data=batch.to_pandas()
        low,high=(251,300) if name=='threshold_validation' else (301,400)
        if not data.step.between(low,high).all():raise ValueError('Datos fuera de ventana')
        start=perf_counter();scores=pipeline.predict_proba(data.loc[:,list(FEATURES)])[:,1]
        seconds+=perf_counter()-start
        rows.append(pd.DataFrame({'row_id':data.row_id,'step':data.step,TARGET:data[TARGET],'score':scores}))
    predictions=pd.concat(rows,ignore_index=True)
    if len(predictions)!=profile['partitions'][name]['rows'] or not predictions.row_id.is_unique:
        raise AssertionError('Conteos de predicción o IDs incorrectos')
    predictions.to_parquet(cache,index=False)
    metadata={**expected,'rows':len(predictions),'inference_seconds':seconds,
              'prediction_cache_reused':False,'predictions_sha256':sha256(cache)}
    manifest.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return predictions,metadata


def evaluate(root:Path=ROOT):
    import mlflow.sklearn
    from mlflow.models import infer_signature
    reports=root/'artifacts/run_reports/evaluation';reports.mkdir(parents=True,exist_ok=True)
    artifacts=root/'artifacts/evaluation';artifacts.mkdir(parents=True,exist_ok=True)
    candidate=json.loads((root/'artifacts/run_reports/optimization/candidate.json').read_text(encoding='utf-8'))
    profile=json.loads((root/'artifacts/run_reports/dataset_profile.json').read_text(encoding='utf-8'))
    path=root/'artifacts/optimization/pipeline_candidate.joblib'
    if sha256(path)!=candidate['model_sha256'] or candidate['fit_step_max']!=250:
        raise ValueError('Candidato distinto del seleccionado')
    protocol_path=reports/'evaluation_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))!=PROTOCOL:
        raise ValueError('Se intentó cambiar el protocolo de evaluación congelado')
    protocol_path.write_text(json.dumps(PROTOCOL,indent=2),encoding='utf-8')
    if (reports/'summary.json').exists():
        summary=json.loads((reports/'summary.json').read_text(encoding='utf-8'))
        if summary['candidate_sha256']!=candidate['model_sha256']:raise ValueError('Evaluación de otro modelo')
        print('Evaluación existente preservada; no se repite para seleccionar otro modelo.',flush=True)
        return summary
    pipeline=joblib.load(path)
    policy_path=reports/'decision_policy.json'
    print('Validando umbral con steps 251–300',flush=True)
    validation,validation_scoring=score_partition(root,'threshold_validation',pipeline,profile,artifacts)
    if policy_path.exists():
        policy=json.loads(policy_path.read_text(encoding='utf-8'))
        if policy['candidate_sha256']!=candidate['model_sha256']:raise ValueError('Política de otro modelo')
        threshold=policy['threshold']
    else:
        threshold,curve=choose_f2_threshold(validation[TARGET],validation.score)
        curve.to_parquet(artifacts/'threshold_candidates.parquet',index=False)
        policy={'threshold':threshold,'criterion':'max_f2','beta':2,'tie_break':'highest_threshold',
                'candidate_sha256':candidate['model_sha256'],'validation_sha256':profile['partitions']['threshold_validation']['parquet_sha256'],
                'validation_rows':len(validation),'validation_steps':[251,300],
                'comparison':'score >= threshold','offline_used_for_selection':False}
        policy_path.write_text(json.dumps(policy,indent=2),encoding='utf-8')
    curve=pd.read_parquet(artifacts/'threshold_candidates.parquet')
    validation_metrics=metrics(validation[TARGET],validation.score,threshold)
    default_validation=metrics(validation[TARGET],validation.score,.5)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for col in ['f2','precision','recall']:axes[0].plot(curve.threshold,curve[col],label=col)
    axes[0].axvline(threshold,color='black',linestyle='--',label=f'Umbral {threshold:.5f}')
    axes[0].set(xlabel='Umbral del score',ylabel='Métrica',title='Selección solo en validación (251–300)',ylim=(0,1.03))
    axes[0].legend(fontsize=8)
    precision,recall,_=precision_recall_curve(validation[TARGET],validation.score)
    axes[1].plot(recall,precision)
    axes[1].scatter(validation_metrics['recall'],validation_metrics['precision'],color='black',label='Decisión elegida')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Curva PR de validación',xlim=(0,1),ylim=(0,1.03))
    axes[1].legend()
    fig.savefig(reports/'threshold_selection.png',dpi=150,bbox_inches='tight');plt.close(fig)
    policy_hash=sha256(policy_path)
    print(f'Política congelada: threshold={threshold:.8f}; F2 validación={validation_metrics["f2"]:.6f}',flush=True)
    # No fitting or selection code below this point.
    print('Evaluación final del bloque offline 301–400',flush=True)
    offline,offline_scoring=score_partition(root,'offline_test',pipeline,profile,artifacts)
    if sha256(policy_path)!=policy_hash:raise AssertionError('La política cambió después de abrir el test')
    offline_metrics=metrics(offline[TARGET],offline.score,threshold)
    default_offline=metrics(offline[TARGET],offline.score,.5)
    (reports/'offline_metrics.json').write_text(json.dumps(offline_metrics,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    precision,recall,_=precision_recall_curve(offline[TARGET],offline.score)
    axes[0].plot(recall,precision,label=f'AP={offline_metrics["average_precision"]:.6f}')
    axes[0].scatter(offline_metrics['recall'],offline_metrics['precision'],color='black',label='Umbral congelado')
    axes[0].set(xlabel='Recall',ylabel='Precision',title='Prueba offline final (301–400)',xlim=(0,1),ylim=(0,1.03));axes[0].legend()
    ConfusionMatrixDisplay(confusion_matrix(offline[TARGET],offline.score>=threshold,labels=[0,1]),display_labels=['Normal','Fraude']).plot(ax=axes[1],colorbar=False,values_format='d')
    axes[1].set_title('Matriz con política congelada')
    fig.savefig(reports/'offline_evaluation.png',dpi=150,bbox_inches='tight');plt.close(fig)
    decision=FrozenThresholdClassifier(pipeline,threshold)
    joblib.dump(decision,artifacts/'decision_model.joblib')
    example=next(pq.ParquetFile(root/'data/processed/development.parquet').iter_batches(batch_size=8,columns=list(FEATURES))).to_pandas()
    for col in example.select_dtypes(include='integer').columns:example[col]=example[col].astype(float)
    np.testing.assert_array_equal(joblib.load(artifacts/'decision_model.joblib').predict(example),decision.predict(example))
    model_dir=artifacts/'mlflow_model'
    if not model_dir.exists():
        mlflow.sklearn.save_model(sk_model=decision,path=str(model_dir),serialization_format='cloudpickle',
            code_paths=[str(root/'src')],signature=infer_signature(example,decision.predict(example)),input_example=example,
            pip_requirements=[f'{p}=={version(p)}' for p in ['mlflow','scikit-learn','pandas','numpy','cloudpickle','joblib','matplotlib','pyarrow','python-dotenv','xgboost']])
    loaded=mlflow.sklearn.load_model(str(model_dir))
    np.testing.assert_array_equal(loaded.predict(example),decision.predict(example))
    np.testing.assert_allclose(loaded.predict_proba(example),decision.predict_proba(example),rtol=1e-7)
    mlflow,exp_id,_,_=configure(root,experiment_name=EXPERIMENT,backend='local')
    with mlflow.start_run(experiment_id=exp_id,run_name='frozen_threshold_offline_evaluation',
                          tags={'stage':'offline_evaluation','production_used':'false','threshold_selection':'validation_only'}) as run:
        mlflow.log_params({'threshold':threshold,'criterion':'max_f2','candidate_sha256':candidate['model_sha256'],
                           'policy_sha256':policy_hash,'fit_step_max':250,'validation_step_min':251,'validation_step_max':300,
                           'offline_step_min':301,'offline_step_max':400})
        mlflow.log_metrics({f'validation_{k}':v for k,v in validation_metrics.items()})
        mlflow.log_metrics({f'offline_{k}':v for k,v in offline_metrics.items()})
        mlflow.log_artifacts(str(model_dir),'model')
        result={'stage':'offline_evaluation','run_id':run.info.run_id,'threshold':threshold,'candidate_sha256':candidate['model_sha256'],
                'policy_sha256':policy_hash,'decision_model_sha256':sha256(artifacts/'decision_model.joblib'),
                'criterion':'max_f2_validation_only','validation_metrics':validation_metrics,'validation_default_05':default_validation,
                'offline_metrics':offline_metrics,'offline_default_05':default_offline,'validation_rows':len(validation),
                'offline_rows':len(offline),'validation_scoring':validation_scoring,'offline_scoring':offline_scoring,
                'model_reload_verified':True,'retraining_performed':False,'production_used':False,
                'source_sha256':profile['source_sha256'],'registry_registration':False,'scores_calibrated':False}
        (reports/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        mlflow.log_artifacts(str(reports),'evaluation_reports')
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':evaluate()
