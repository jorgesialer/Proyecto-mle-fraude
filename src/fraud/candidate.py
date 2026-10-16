"""Fit the CV-selected candidate on development; do not open reserved periods."""
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .boosting import BoostingEncoder, training_weight
from .modeling import FeatureContract
from .optimize import EXPERIMENT, fixed_params, prepare_matrix
from .prepare import ROOT, sha256
from .schema import FEATURES
from .tracking import configure


def fit_candidate(root:Path=ROOT):
    import mlflow.sklearn
    from mlflow.models import infer_signature
    reports=root/'reports/optimization'
    result_path=reports/'candidate.json'
    artifacts=root/'artifacts/optimization'
    summary=json.loads((reports/'summary.json').read_text(encoding='utf-8'))
    profile=json.loads((root/'reports/dataset_profile.json').read_text(encoding='utf-8'))
    if result_path.exists():
        previous=json.loads(result_path.read_text(encoding='utf-8'))
        if sha256(artifacts/'pipeline_candidate.joblib')!=previous['model_sha256']:
            raise ValueError('El modelo no coincide con la evidencia guardada')
        return previous
    example=next(pq.ParquetFile(root/'data/processed/development.parquet').iter_batches(batch_size=8,columns=list(FEATURES))).to_pandas()
    selected=next(s for s in summary['methods'] if s['method']==summary['best_boosting_method'])
    mlflow,exp_id,_,_=configure(root,experiment_name=EXPERIMENT,backend='local')
    with mlflow.start_run(experiment_id=exp_id,run_name='candidate_development_refit',
                          tags={'stage':'candidate_refit','role':'not_final_champion','offline_test_used':'false','production_used':'false'}) as run:
        start=perf_counter()
        if summary['selected_family']=='xgboost':
            data=prepare_matrix(root,profile)
            plan=json.loads((reports/'search_plan.json').read_text(encoding='utf-8'))
            weight=training_weight(data['y'])
            model=XGBClassifier(**fixed_params(plan['fixed_params']['n_jobs']),**selected['best_params'],scale_pos_weight=weight)
            print('Ajustando XGBoost seleccionado en todos los steps 1–250',flush=True)
            model.fit(data['X'],data['y'])
            pipeline=Pipeline([('contract',FeatureContract().fit(example)),
                               ('encoder',BoostingEncoder().fit(example)),('model',model)])
            model.save_model(artifacts/'candidate_booster.ubj')
            params={**selected['best_params'],**fixed_params(plan['fixed_params']['n_jobs']),'scale_pos_weight':weight}
            selection_method=selected['method']
        else:
            pipeline=joblib.load(root/'artifacts/baseline/pipeline_baseline.joblib')
            params=json.loads((root/'reports/baseline/summary.json').read_text(encoding='utf-8'))['tree_parameters']
            selection_method='baseline_retained'
        elapsed=perf_counter()-start
        model_path=artifacts/'candidate_mlflow_model'
        if model_path.exists():raise FileExistsError('Preservar el artefacto anterior antes de repetir')
        joblib.dump(pipeline,artifacts/'pipeline_candidate.joblib')
        example=example.copy()
        for col in example.select_dtypes(include='integer').columns:example[col]=example[col].astype(float)
        packages=['mlflow','scikit-learn','pandas','numpy','cloudpickle','joblib','matplotlib','pyarrow','python-dotenv','xgboost']
        mlflow.sklearn.save_model(sk_model=pipeline,path=str(model_path),serialization_format='cloudpickle',
            code_paths=[str(root/'src')],signature=infer_signature(example,pipeline.predict(example)),
            input_example=example,pip_requirements=[f'{p}=={version(p)}' for p in packages])
        loaded=mlflow.sklearn.load_model(str(model_path))
        np.testing.assert_allclose(loaded.predict_proba(example),pipeline.predict_proba(example),rtol=1e-7)
        mlflow.log_params({'family':summary['selected_family'],'selection_method':selection_method,
                           'training_rows':profile['partitions']['development']['rows'],**params})
        mlflow.log_metric('refit_seconds',elapsed)
        mlflow.log_artifacts(str(model_path),'model')
        result={'run_id':run.info.run_id,'family':summary['selected_family'],'selection_method':selection_method,
                'params':params,'fit_step_min':1,'fit_step_max':250,'refit_seconds':elapsed,
                'model_sha256':sha256(artifacts/'pipeline_candidate.joblib'),'model_reload_verified':True,
                'threshold_selected':False,'offline_test_used':False,'production_used':False,
                'model_role':'candidate_before_threshold_and_offline_test','source_sha256':profile['source_sha256']}
        result_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        mlflow.log_artifact(str(result_path))
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':fit_candidate()
