"""Execute the frozen March study and generate public aggregate results."""
import hashlib
import platform
import importlib.metadata
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import ctr_study as s

def run():
    frame=s.make_frame()
    frame['fold']=frame.client_hash_id.map(lambda x:int(hashlib.sha256(('ctr-v1:'+x).encode()).hexdigest()[:8],16)%5)
    data=frame.loc[frame.evaluation_eligible].copy()
    train=data.loc[data.fold>=2].copy()
    valid=data.loc[data.fold==1].copy()
    test=data.loc[data.fold==0].copy()
    assert all(len(x)>0 for x in (train,valid,test))
    groups=[set(x.client_hash_id) for x in (train,valid,test)]
    assert not any(groups[i]&groups[j] for i,j in [(0,1),(0,2),(1,2)])
    assert set(s.FEATURES).isdisjoint({'future_ctr','future_impressions','client_hash_id','content_hash_id'})
    models={
        'Ridge':make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
        'Gradient boosting':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,
            min_samples_leaf=40,l2_regularization=1.0,early_stopping=False,random_state=s.SEED)
    }
    # No model fitting or reference medians use validation/test clients.
    reference=frame.loc[frame.fold>=2]
    band=reference.groupby('position_band',observed=True).past_ctr.median()
    def reference_for(f):
        return f.position_band.astype('string').map(band).astype(float).fillna(reference.past_ctr.median()).to_numpy()
    for model in models.values(): model.fit(train[s.FEATURES],train.future_ctr)
    def predictions(f):
        return {'Persistence':f.past_ctr.to_numpy(),'Position reference':reference_for(f),
                **{name:np.clip(m.predict(f[s.FEATURES]),0,100) for name,m in models.items()}}
    pv=predictions(valid)
    validation=[s.metric_row(valid.future_ctr,p,valid,name) for name,p in pv.items()]
    # Selection rule fixed before test: lowest validation MAE; baseline is eligible to win.
    selected=min(validation,key=lambda r:r['mae_pp'])['method']
    pt=predictions(test)
    test_metrics=[s.metric_row(test.future_ctr,p,test,name) for name,p in pt.items()]
    test_ref=reference_for(test)
    proxy=(test.future_ctr.to_numpy()<test_ref).astype(int)
    ranking=[]
    for name in ['Persistence','Ridge','Gradient boosting']:
        scores=np.maximum(test_ref-pt[name],0)/100*test.past_impressions.to_numpy()
        order=np.lexsort((test.content_hash_id.to_numpy(),-scores))
        for k in (10,50):
            top=order[:k]
            ranking.append({'method':name,'k':k,'future_below_reference_fraction':float(proxy[top].mean()),
                            'base_rate':float(proxy.mean()),'positive_score_items':int((scores[top]>0).sum())})
    # Paired uncertainty: each client contributes equally; resample clients, not correlated pages.
    delta=np.abs(test.future_ctr.to_numpy()-pt[selected])-np.abs(test.future_ctr.to_numpy()-pt['Persistence'])
    client_delta=pd.Series(delta,index=test.index).groupby(test.client_hash_id).mean()
    rng=np.random.default_rng(s.SEED)
    draws=rng.choice(client_delta.to_numpy(),size=(2000,len(client_delta)),replace=True).mean(axis=1)
    interval=np.quantile(draws,[0.025,0.975]).tolist()
    # Importance on development validation only, for the fixed boosted candidate.
    perm=permutation_importance(models['Gradient boosting'],valid[s.FEATURES],valid.future_ctr,
        scoring='neg_mean_absolute_error',n_repeats=5,random_state=s.SEED,n_jobs=1)
    importance=pd.DataFrame({'feature':s.FEATURES,'mae_increase_pp':perm.importances_mean,
                             'std_pp':perm.importances_std}).sort_values('mae_increase_pp',ascending=False)
    err=test.copy()
    err['prediction']=pt[selected]
    err['abs_error_pp']=np.abs(err.future_ctr-err.prediction)
    grouped=err.groupby('position_band',observed=True).agg(n=('abs_error_pp','size'),mae_pp=('abs_error_pp','mean'))
    worst=err.nlargest(3,'abs_error_pp')[['content_hash_id','past_ctr','future_ctr','prediction','abs_error_pp','mean_position']]
    queue=s.baseline_queue(frame,reference)
    # First-half review queue; no future availability/outcome filter.
    confidence_limits=[
        'Large exposure can magnify a small gap; check query mix before editing.',
        'Search-result answers may explain low CTR without a snippet defect.',
        'A broad position band does not control device and country mix.',
        'Daily average rank can hide query-level position changes.',
        'Reference-client medians may not transfer to this content niche.',
        'Informational intent may justify low CTR.',
        'A few highly exposed queries can dominate the aggregate.',
        'A short event can dominate this 15-day window.',
        'Changing an appropriate title can reduce performance.',
        'Measurement coverage does not guarantee comparable intent.'
    ]
    top=[]
    for i,(_,r) in enumerate(queue.head(10).iterrows()):
        top.append({'rank':i+1,'content_id':r.content_hash_id,'action':r.action,
                    'evidence':f"CTR {r.past_ctr:.3f}% vs reference {r.reference_ctr:.3f}%; {r.past_impressions:,.0f} impressions; gap score {r.score:.1f}.",
                    'caveat':confidence_limits[i]})
    sensitivity=[]
    for threshold in (500,1000,3000):
        mask=test.future_impressions.to_numpy()>=threshold
        if mask.any():
            sensitivity.append({'outcome_impression_threshold':threshold,'pages':int(mask.sum()),
                'selected_mae_pp':float(mean_absolute_error(test.future_ctr.to_numpy()[mask],pt[selected][mask])),
                'persistence_mae_pp':float(mean_absolute_error(test.future_ctr.to_numpy()[mask],pt['Persistence'][mask]))})
    figdir=s.ROOT/'work/figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    names=[r['method'] for r in test_metrics]; values=[r['mae_pp'] for r in test_metrics]
    ax.barh(names,values,color=['#526c83','#adc2c5','#487b67','#c68a36'])
    for i,x in enumerate(values): ax.text(x+0.003,i,f'{x:.4f}',va='center')
    ax.set_xlim(0,max(values)*1.25); ax.set_xlabel('Mean absolute error (CTR percentage points; lower is better)')
    ax.set_title('Held-out clients: March 16–31 CTR'); fig.savefig(figdir/'test_error.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.barh(importance.feature.iloc[::-1],importance.mae_increase_pp.iloc[::-1],color='#487b67')
    ax.set_xlabel('Validation MAE increase after shuffling (percentage points)')
    ax.set_title('Boosted model: development-client permutation importance')
    fig.savefig(figdir/'feature_importance.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.bar(grouped.index.astype(str),grouped.mae_pp,color='#526c83')
    for i,(_,r) in enumerate(grouped.iterrows()):ax.text(i,r.mae_pp,f'n={int(r.n):,}',ha='center',va='bottom')
    ax.set_ylim(0,grouped.mae_pp.max()*1.25);ax.set_xlabel('First-half mean-position band');ax.set_ylabel('Selected-method MAE (pp)')
    ax.set_title('Error varies by position on held-out clients');fig.savefig(figdir/'error_by_position.png',dpi=160);plt.close(fig)
    metadata={'study':'CTR opportunity scoring: March 2026 warehouse slice','seed':s.SEED,
        'data_sha256':hashlib.sha256((s.DATA/'march_2026.parquet').read_bytes()).hexdigest(),
        'raw_rows':9841378,'feature_pages':len(frame),'evaluation_pages':len(data),'excluded_from_evaluation':len(frame)-len(data),
        'split':[{ 'partition':name,'pages':len(f),'clients':int(f.client_hash_id.nunique())} for name,f in [('train',train),('validation',valid),('test',test)]],
        'selected_on_validation':selected,'validation':validation,'test':test_metrics,'ranking_proxy':ranking,
        'selected_minus_persistence_client_macro_mae_ci95_pp':interval,
        'paired_client_macro_difference_pp':float(client_delta.mean()),
        'importance':importance.to_dict('records'),
        'ridge_standardized_coefficients':dict(zip(s.FEATURES,models['Ridge'].named_steps['ridge'].coef_.tolist())),
        'errors_by_position':grouped.reset_index().to_dict('records'),
        'largest_errors':worst.round(5).to_dict('records'),'sensitivity':sensitivity,
        'top10':top,'queue_pages':len(queue),'independent_actionability_labels':0,
        'versions':{p:importlib.metadata.version(p) for p in ['duckdb','pandas','numpy','scikit-learn','matplotlib']},
        'python':platform.python_version()}
    s.save_json('capstone_metrics.json',metadata)
    return metadata

if __name__=='__main__':
    result=run()
    print(json.dumps({k:result[k] for k in ['split','selected_on_validation','validation','test','selected_minus_persistence_client_macro_mae_ci95_pp']},indent=2))
