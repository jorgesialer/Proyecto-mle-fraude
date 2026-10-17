"""Post-evaluation explanations; never used to select or refit the model."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from xgboost import DMatrix
from .boosting import NUMERIC
from .prepare import ROOT, sha256
from .schema import FEATURES, TARGET
from .tracking import configure


def plot_response(pipeline,X,y,path):
    """Population PDP plus minority-enriched ICE; different populations are explicit."""
    population=X.sample(n=min(2000,len(X)),random_state=42)
    minority=X.loc[np.asarray(y)==1].head(16)
    negatives=X.loc[np.asarray(y)==0].sample(n=min(24,int((np.asarray(y)==0).sum())),random_state=42)
    ice=pd.concat([minority,negatives],ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,ax in zip(['amount','oldbalanceOrg'],axes):
        grid=np.unique(np.quantile(X[name].dropna(),np.linspace(.01,.99,25)))
        means=[];curves=[]
        for value in grid:
            changed=population.copy();changed[name]=value
            means.append(pipeline.predict_proba(changed)[:,1].mean())
            changed=ice.copy();changed[name]=value
            curves.append(pipeline.predict_proba(changed)[:,1])
        curves=np.asarray(curves)
        for i in range(len(ice)):
            ax.plot(grid,curves[:,i],color='#bb4430' if i<len(minority) else '#176b87',alpha=.25,lw=.8)
        ax.plot(grid,means,color='black',ls='--',lw=2,label='PDP: muestra uniforme de 2.000')
        ax.set(xlabel=name,ylabel='Score (escala symlog)',yscale='symlog',title=f'ICE: {len(minority)} fraudes (rojo) + {len(negatives)} no fraudes')
        ax.set_yscale('symlog',linthresh=.001);ax.legend(fontsize=8)
    fig.suptitle('PDP / ICE del score · asociaciones, no efectos causales');fig.tight_layout()
    fig.savefig(path,dpi=145);plt.close(fig)


def sample_rows(path, n=20_000, seed=42):
    """Uniform sample without replacement, independent of class and score."""
    source=pq.ParquetFile(path)
    positions=np.sort(np.random.default_rng(seed).choice(source.metadata.num_rows,
                       min(n,source.metadata.num_rows),replace=False))
    result=[]; offset=0
    for batch in source.iter_batches(batch_size=100_000):
        selected=positions[(positions>=offset)&(positions<offset+len(batch))]-offset
        if len(selected):result.append(batch.take(selected).to_pandas())
        offset+=len(batch)
    return pd.concat(result,ignore_index=True)


def contributions(pipeline, X):
    encoded=pipeline[:-1].transform(X)
    booster=pipeline[-1].get_booster()
    matrix=DMatrix(encoded)
    values=booster.predict(matrix,pred_contribs=True)
    margin=booster.predict(matrix,output_margin=True)
    add_error=float(np.max(np.abs(values.sum(axis=1)-margin)))
    score_error=float(np.max(np.abs(expit(margin)-pipeline.predict_proba(X)[:,1])))
    if add_error>2e-5 or score_error>1e-6:raise AssertionError('TreeSHAP no reproduce la predicción')
    # Group one-hot type before computing absolute global importance.
    grouped=np.column_stack([values[:,:len(NUMERIC)],values[:,len(NUMERIC):-1].sum(axis=1)])
    return grouped,values[:,-1],margin,add_error,score_error


def explain(root:Path=ROOT):
    out=root/'artifacts/run_reports/interpretability';out.mkdir(parents=True,exist_ok=True)
    evaluation=json.loads((root/'artifacts/run_reports/evaluation/summary.json').read_text(encoding='utf-8'))
    model_path=root/'artifacts/evaluation/decision_model.joblib'
    if sha256(model_path)!=evaluation['decision_model_sha256']:raise ValueError('Modelo diferente')
    if (out/'summary.json').exists():
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        if summary['model_sha256']!=sha256(model_path):raise ValueError('Explicaciones de otro modelo')
        return summary
    decision=joblib.load(model_path);pipeline=decision.pipeline
    sample=sample_rows(root/'data/processed/offline_test.parquet')
    X=sample.loc[:,list(FEATURES)]; y=sample[TARGET]
    print(f'TreeSHAP y permutación: {len(X)} filas, {int(y.sum())} fraudes',flush=True)
    shap_values,bias,margin,add_error,score_error=contributions(pipeline,X)
    names=[*NUMERIC,'type']
    importance=pd.DataFrame({'feature':names,'mean_abs_shap_log_odds':np.abs(shap_values).mean(axis=0)})
    importance=importance.sort_values('mean_abs_shap_log_odds',ascending=False)
    importance.to_csv(out/'global_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,6)); ordered=importance.iloc[::-1]
    ax.barh(ordered.feature,ordered.mean_abs_shap_log_odds,color='#176b87')
    ax.set(xlabel='Media |TreeSHAP| (log-odds)',title='Importancia global · muestra uniforme de test offline')
    fig.tight_layout();fig.savefig(out/'shap_global.png',dpi=145);plt.close(fig)
    rng=np.random.default_rng(42);fig,ax=plt.subplots(figsize=(11,7))
    for j,name in enumerate(ordered.feature):
        k=names.index(name); positions=rng.choice(len(X),min(3000,len(X)),replace=False)
        color=np.full(len(positions),.5)
        if name!='type':
            raw=X[name].to_numpy(dtype=float)[positions]; finite=np.isfinite(raw)
            if finite.any():
                lo,hi=np.quantile(raw[finite],[.01,.99]);color[finite]=np.clip((raw[finite]-lo)/max(hi-lo,1e-12),0,1)
        dots=ax.scatter(shap_values[positions,k],j+rng.normal(0,.12,len(positions)),c=color,cmap='coolwarm',vmin=0,vmax=1,s=4,alpha=.45)
    ax.set(yticks=range(len(ordered)),yticklabels=ordered.feature,xlabel='Contribución TreeSHAP (log-odds)',title='Distribución de contribuciones · color relativo por variable')
    ax.axvline(0,color='gray',lw=.8);fig.colorbar(dots,ax=ax,label='Valor: bajo (azul) / alto (rojo); type y NaN neutros')
    fig.tight_layout();fig.savefig(out/'shap_distribution.png',dpi=145);plt.close(fig)
    perm=permutation_importance(pipeline,X,y,scoring='average_precision',n_repeats=3,random_state=42,n_jobs=1)
    permutation=pd.DataFrame({'feature':FEATURES,'ap_decrease_mean':perm.importances_mean,'ap_decrease_std':perm.importances_std}).sort_values('ap_decrease_mean',ascending=False)
    permutation.to_csv(out/'permutation_importance.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,6));p=permutation.iloc[::-1]
    ax.barh(p.feature,p.ap_decrease_mean,xerr=p.ap_decrease_std,color='#287d66');ax.set(xlabel='Disminución AP ± DE de 3 permutaciones',title=f'Permutación · {len(X):,} filas, {int(y.sum())} fraudes')
    fig.tight_layout();fig.savefig(out/'permutation.png',dpi=145);plt.close(fig)
    # Brute-force response of the fixed pipeline. Correlated derived features limit interpretation.
    plot_response(pipeline,X,y,out/'pdp_ice.png')
    predictions=pd.read_parquet(root/'artifacts/evaluation/offline_test_predictions.parquet')
    predictions['predicted']=(predictions.score>=decision.threshold).astype(int)
    selections=[]
    for label,true,pred in [('TP',1,1),('FP',0,1),('FN',1,0),('TN',0,0)]:
        cases=predictions[(predictions[TARGET]==true)&(predictions.predicted==pred)].head(2).copy()
        cases['case']=label;selections.append(cases)
    selected=pd.concat(selections,ignore_index=True)
    cases=pd.read_parquet(root/'data/processed/offline_test.parquet',filters=[('row_id','in',selected.row_id.tolist())])
    cases=cases.set_index('row_id').loc[selected.row_id].reset_index()
    local,base,raw,local_add,local_score=contributions(pipeline,cases.loc[:,list(FEATURES)])
    local_table=[];fig,axes=plt.subplots(2,2,figsize=(14,10))
    for i,row in selected.iterrows():
        local_table.append({'row_id':int(row.row_id),'step':int(row.step),'case':row['case'],'actual':int(row[TARGET]),
            'predicted':int(row.predicted),'score':float(row.score),'threshold':decision.threshold,
            'base_log_odds':float(base[i]),'margin_log_odds':float(raw[i]),
            'contributions_log_odds':dict(zip(names,map(float,local[i])))})
        if i%2==0:
            ax=axes.flat[i//2];idx=np.argsort(np.abs(local[i]))[-8:]
            ax.barh(np.array(names)[idx],local[i,idx],color=['#bb4430' if v>0 else '#176b87' for v in local[i,idx]])
            ax.set(title=f'{row["case"]} · row {int(row.row_id)} · score {row.score:.5f}',xlabel=f'Contribución log-odds (base {base[i]:.2f})');ax.axvline(0,color='gray',lw=.8)
    fig.tight_layout();fig.savefig(out/'shap_local_cases.png',dpi=145);plt.close(fig)
    (out/'local_cases.json').write_text(json.dumps(local_table,indent=2),encoding='utf-8')
    mlflow,exp_id,_,_=configure(root,experiment_name='fraud-interpretability',backend='local')
    with mlflow.start_run(experiment_id=exp_id,run_name='frozen_offline_explanations') as run:
        mlflow.log_params({'method':'native_xgboost_TreeSHAP','sample_rows':len(X),'sample_frauds':int(y.sum()),'seed':42,'permutation_repeats':3,'model_sha256':sha256(model_path)})
        mlflow.log_metrics({'additivity_max_abs_error':max(add_error,local_add),'score_parity_max_abs_error':max(score_error,local_score)})
        mlflow.log_artifacts(str(out),artifact_path='interpretability')
        summary={'model_sha256':sha256(model_path),'run_id':run.info.run_id,'rows':len(X),'frauds':int(y.sum()),
            'method':'native XGBoost TreeSHAP (tree-path-dependent; no background dataset)',
            'additivity_max_abs_error':max(add_error,local_add),'score_parity_max_abs_error':max(score_error,local_score),
            'top_features':importance.feature.head(5).tolist(),'local_cases':len(local_table),'production_used':False,
            'limitations':['Sample uniform; few positive examples make permutation AP variable.',
            'Correlated original/derived features complicate permutation and PDP/ICE; perturbations may be unrealistic.',
            'SHAP contributions add in raw log-odds, not probability; scores are not calibrated.',
            'No protected demographic attributes: no demographic fairness claim. Explanations are not causal.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':print(json.dumps(explain(),indent=2))
