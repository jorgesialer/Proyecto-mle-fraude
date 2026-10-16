"""Historical production replay with a frozen model, explicit alerts and labels."""
import json
from pathlib import Path
from time import perf_counter
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from .baseline import metrics
from .explain import sample_rows
from .prepare import ROOT, sha256
from .schema import FEATURES, TARGET
from .tracking import configure

POLICY={'reference_steps':[301,400],'current_steps':[401,743],'window_steps':50,
 'reference_sample_rows':50000,'seed':42,'psi_alert':0.2,'ks_min_effect':0.1,'bh_alpha':0.05,
 'recall_floor':0.95,'precision_floor':0.90,'minimum_frauds_for_performance_alert':30,
 'adwin_delta':0.002,'label_availability':'immediate historical ground truth; simulated, not live',
 'automatic_retraining':False,'threshold_change':False}


def psi(reference,current,categorical=False):
    """Reference-only bins plus explicit missing bucket; additive smoothing."""
    r=pd.Series(reference);c=pd.Series(current)
    if categorical:
        categories=sorted(set(r.dropna())|set(c.dropna()),key=str)
        rc=np.array([(r==v).sum() for v in categories]+[r.isna().sum()],dtype=float)
        cc=np.array([(c==v).sum() for v in categories]+[c.isna().sum()],dtype=float)
    else:
        rv=r.dropna().to_numpy(dtype=float);cv=c.dropna().to_numpy(dtype=float)
        if not len(rv):raise ValueError('Referencia sin valores numéricos')
        inner=np.unique(np.quantile(rv,np.linspace(0,1,11)[1:-1]))
        bins=np.r_[-np.inf,inner,np.inf]
        rc=np.r_[np.histogram(rv,bins=bins)[0],r.isna().sum()].astype(float)
        cc=np.r_[np.histogram(cv,bins=bins)[0],c.isna().sum()].astype(float)
    rp=(rc+.5)/(rc.sum()+.5*len(rc));cp=(cc+.5)/(cc.sum()+.5*len(cc))
    return float(np.sum((cp-rp)*np.log(cp/rp)))


def bh_reject(p_values,alpha=.05):
    values=np.asarray(p_values,dtype=float);n=len(values)
    order=np.argsort(values);passes=values[order]<=alpha*np.arange(1,n+1)/n
    result=np.zeros(n,dtype=bool)
    if passes.any():result[order[:np.flatnonzero(passes)[-1]+1]]=True
    return result


def detect_changes(errors,delta=.002):
    from river.drift import ADWIN
    detector=ADWIN(delta=delta);signals=[]
    for i,error in enumerate(errors):
        detector.update(int(error))
        if detector.drift_detected:signals.append(i)
    return signals


def monitor(root:Path=ROOT):
    out=root/'reports/monitoring';out.mkdir(parents=True,exist_ok=True)
    artifacts=root/'artifacts/monitoring';artifacts.mkdir(parents=True,exist_ok=True)
    evaluation=json.loads((root/'reports/evaluation/summary.json').read_text(encoding='utf-8'))
    model_path=root/'artifacts/evaluation/decision_model.joblib';model_hash=sha256(model_path)
    if model_hash!=evaluation['decision_model_sha256']:raise ValueError('Modelo diferente del evaluado')
    if (out/'summary.json').exists():
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        if summary['model_sha256']!=model_hash:raise ValueError('Monitoreo de otro modelo')
        return summary
    policy_path=out/'alert_policy.json'
    if policy_path.exists() and json.loads(policy_path.read_text(encoding='utf-8'))!=POLICY:
        raise ValueError('Política de alertas modificada después de iniciar replay')
    policy_path.write_text(json.dumps(POLICY,indent=2),encoding='utf-8')
    policy_hash=sha256(policy_path)
    profile=json.loads((root/'reports/dataset_profile.json').read_text(encoding='utf-8'))
    prod_path=root/'data/processed/production.parquet'
    if sha256(prod_path)!=profile['partitions']['production']['parquet_sha256']:raise ValueError('Producción modificada')
    model=joblib.load(model_path)
    ref=sample_rows(root/'data/processed/offline_test.parquet',n=50_000)
    ref['score']=model.predict_proba(ref.loc[:,list(FEATURES)])[:,1]
    pieces=[];seconds=0.
    for batch in pq.ParquetFile(prod_path).iter_batches(batch_size=100_000):
        data=batch.to_pandas()
        if not data.step.between(401,743).all():raise ValueError('Ventana productiva incorrecta')
        start=perf_counter();data['score']=model.predict_proba(data.loc[:,list(FEATURES)])[:,1]
        seconds+=perf_counter()-start;pieces.append(data)
    current=pd.concat(pieces,ignore_index=True).sort_values(['step','row_id']).reset_index(drop=True)
    if len(current)!=profile['partitions']['production']['rows'] or not current.row_id.is_unique:raise AssertionError('Replay incompleto')
    current['predicted']=(current.score>=model.threshold).astype('int8')
    current['window']=((current.step-401)//50).astype(int)
    current[['row_id','step',TARGET,'score','predicted','window']].to_parquet(artifacts/'production_predictions.parquet',index=False)
    windows=[];drift_rows=[];quality=[]
    for window,data in current.groupby('window',sort=True):
        print(f'Monitoreo steps {int(data.step.min())}–{int(data.step.max())}: {len(data):,} filas',flush=True)
        result=metrics(data[TARGET],data.score,model.threshold)
        windows.append({'window':int(window),'step_min':int(data.step.min()),'step_max':int(data.step.max()),'rows':len(data),'frauds':int(data[TARGET].sum()),**result})
        group=[]
        for feature in [*FEATURES,'score']:
            categorical=feature=='type'
            distance=psi(ref[feature],data[feature],categorical=categorical)
            ks,p=(np.nan,np.nan) if categorical else ks_2samp(ref[feature].dropna(),data[feature].dropna(),method='asymp')
            group.append({'window':int(window),'feature':feature,'psi':distance,'ks_statistic':float(ks),'ks_pvalue':float(p)})
            quality.append({'window':int(window),'feature':feature,'missing_rate':float(data[feature].isna().mean()),
                'reference_missing_rate':float(ref[feature].isna().mean()),'zero_rate':None if categorical else float(data[feature].eq(0).mean())})
        numeric=[r for r in group if np.isfinite(r['ks_pvalue'])]
        rejected=bh_reject([r['ks_pvalue'] for r in numeric])
        for row,flag in zip(numeric,rejected):row['ks_bh_reject']=bool(flag)
        for row in group:
            row['drift_alert']=bool(row['psi']>=POLICY['psi_alert'] or (row.get('ks_bh_reject',False) and row['ks_statistic']>=POLICY['ks_min_effect']))
        drift_rows.extend(group)
    windows=pd.DataFrame(windows);drift=pd.DataFrame(drift_rows);quality=pd.DataFrame(quality)
    windows['drifting_columns']=windows.window.map(drift.groupby('window').drift_alert.sum())
    windows['performance_alert']=(windows.frauds>=30)&((windows.recall<.95)|(windows.precision<.90))
    windows.to_csv(out/'windows.csv',index=False);drift.to_csv(out/'feature_drift.csv',index=False);quality.to_csv(out/'quality.csv',index=False)
    errors=current.predicted.to_numpy()!=current[TARGET].to_numpy()
    observed=detect_changes(errors)
    # Deliberate conditional label corruption for a separate detector sensitivity demonstration.
    stress_y=current[TARGET].to_numpy().copy();mid=len(stress_y)//2
    flip=np.random.default_rng(42).random(len(stress_y)-mid)<.15
    stress_y[mid:]=np.where(flip,1-stress_y[mid:],stress_y[mid:])
    stress=detect_changes(current.predicted.to_numpy()!=stress_y)
    signals={'observed_error_changes':[{'position':i,'step':int(current.step.iloc[i]),'row_id':int(current.row_id.iloc[i])} for i in observed],
        'controlled_scenario':{'description':'Synthetic ONLY: flip 15% of labels after midpoint, seed42; original data/model unchanged.',
            'change_position':mid,'flipped_labels':int(flip.sum()),'detections':stress,
            'first_detection_delay_rows':next((i-mid for i in stress if i>=mid),None)},
        'interpretation':'Observed changes in error mean are investigation signals, not proof of conditional concept drift. No alert is not proof of stationarity.'}
    (out/'adwin.json').write_text(json.dumps(signals,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(13,9));x=windows.step_max
    axes[0,0].plot(x,windows.precision,'o-',label='Precisión');axes[0,0].plot(x,windows.recall,'o-',label='Recall');axes[0,0].plot(x,windows.f2,'o-',label='F2');axes[0,0].set_ylim(0,1.03);axes[0,0].legend()
    axes[0,0].set_title('Desempeño con etiquetas históricas disponibles')
    axes[0,1].plot(x,windows.prevalence*100,'o-',label='Fraude real');axes[0,1].plot(x,windows.alert_rate*100,'o-',label='Alertas');axes[0,1].legend();axes[0,1].set(ylabel='Porcentaje',title='Prevalencia y carga de alertas')
    axes[1,0].bar(x,windows.rows,width=30,color='#176b87');axes[1,0].set_title('Volumen por ventana (última ventana parcial)')
    for name in ['amount','origin_debit_discrepancy','score','type']:
        d=drift[drift.feature==name];axes[1,1].plot(x,d.psi,'o-',label=name)
    axes[1,1].axhline(.2,color='red',ls='--',label='PSI 0,2');axes[1,1].legend(fontsize=8);axes[1,1].set_title('PSI respecto a muestra offline fija')
    for ax in axes.flat:ax.set_xlabel('Último step de ventana');ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'production_monitoring.png',dpi=145);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,7));heat=drift.pivot(index='feature',columns='window',values='psi')
    plot=ax.imshow(heat.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0,vmax=max(.4,float(heat.max().max())))
    ax.set(yticks=range(len(heat)),yticklabels=heat.index,xticks=range(len(heat.columns)),xticklabels=[f'{401+int(i)*50}+' for i in heat.columns],title='PSI por variable y ventana · referencia steps 301–400');fig.colorbar(plot,ax=ax)
    fig.tight_layout();fig.savefig(out/'drift_heatmap.png',dpi=145);plt.close(fig)
    # Evidently is an additional standard HTML report, using explicit feature roles.
    import os
    os.environ['DO_NOT_TRACK']='1'
    from evidently import Dataset,DataDefinition,Report
    from evidently.presets import DataDriftPreset
    production_sample=current.sample(n=min(50_000,len(current)),random_state=42)
    cols=[*FEATURES,'score']
    definition=DataDefinition(numerical_columns=[c for c in cols if c!='type'],categorical_columns=['type'])
    snapshot=Report([DataDriftPreset()]).run(
        current_data=Dataset.from_pandas(production_sample[cols],data_definition=definition),
        reference_data=Dataset.from_pandas(ref[cols],data_definition=definition))
    snapshot.save_html(str(out/'evidently_drift.html'))
    snapshot.save_json(str(out/'evidently_drift.json'))
    overall=metrics(current[TARGET],current.score,model.threshold)
    summary={'model_sha256':model_hash,'policy_sha256':policy_hash,'threshold':model.threshold,'rows':len(current),
        'frauds':int(current[TARGET].sum()),'steps':[int(current.step.min()),int(current.step.max())],
        'metrics':overall,'reference_rows':len(ref),'reference_frauds':int(ref[TARGET].sum()),
        'reference_steps':[301,400],'inference_seconds':seconds,'inference_ms_per_1000_rows':seconds/len(current)*1e6,
        'windows':len(windows),'windows_with_drift':int((windows.drifting_columns>0).sum()),
        'performance_alert_windows':int(windows.performance_alert.sum()),'observed_adwin_signals':len(observed),
        'stress_detection_delay_rows':signals['controlled_scenario']['first_detection_delay_rows'],
        'production_predictions_sha256':sha256(artifacts/'production_predictions.parquet'),
        'retraining_performed':False,'deployment':'historical local replay; no live deployment',
        'evidently_sample_rows':len(production_sample),'evidently_default_tests_separate_from_custom_alert_policy':True}
    if sha256(policy_path)!=policy_hash or sha256(model_path)!=model_hash:raise AssertionError('Modelo/política cambiaron')
    mlflow,exp_id,_,_=configure(root,experiment_name='fraud-monitoring',backend='local')
    with mlflow.start_run(experiment_id=exp_id,run_name='frozen_production_replay') as run:
        mlflow.log_params({'model_sha256':model_hash,'policy_sha256':policy_hash,'threshold':model.threshold,'label_mode':'historical_immediate'})
        mlflow.log_metrics({**overall,'rows':len(current),'observed_adwin_signals':len(observed),'inference_seconds':seconds})
        mlflow.log_artifacts(str(out),artifact_path='monitoring');summary['run_id']=run.info.run_id
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':print(json.dumps(monitor(),indent=2))
