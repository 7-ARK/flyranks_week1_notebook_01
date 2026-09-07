"""Reproducible CTR decision-support study on approved local warehouse partitions."""
from pathlib import Path
import os
import hashlib
import json
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('FLYRANK_DATA_DIR', ROOT / 'data' / 'warehouse'))
OUT = ROOT / 'work' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ['past_ctr', 'log_impressions', 'mean_position', 'impression_cv', 'zero_click_fraction']
SEED = 42

def connection(month='march'):
    filename = {'march':'march_2026.parquet', 'june':'june_2026.parquet'}[month]
    path = DATA / filename
    if not path.is_file():
        raise FileNotFoundError(f'Approved warehouse file required: {path}. See work/REPRODUCE.md.')
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='2GB'")
    con.read_parquet(str(path)).create_view('daily')
    return con

VERIFICATION_QUERIES = {
    'grain': '''SELECT COUNT(*) AS duplicate_grain_groups FROM (
        SELECT report_date, client_hash_id, content_hash_id, COUNT(*) n
        FROM daily GROUP BY 1,2,3 HAVING COUNT(*) > 1)''',
    'count_and_span': '''SELECT COUNT(*) AS rows, MIN(report_date) AS first_date,
        MAX(report_date) AS last_date, COUNT(DISTINCT client_hash_id) AS clients,
        COUNT(DISTINCT content_hash_id) AS content_items,
        COUNT(*) FILTER (WHERE report_date IS NULL OR client_hash_id IS NULL
          OR content_hash_id IS NULL) AS missing_keys FROM daily''',
    'availability': '''SELECT COUNT(*) AS all_rows,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE) AS gsc_available,
        COUNT(*) FILTER (WHERE ga4_data_available IS TRUE) AS ga4_available,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE AND ga4_data_available IS TRUE) AS both_available,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE AND
          (gsc_impressions IS NULL OR gsc_clicks IS NULL)) AS missing_search_counts,
        COUNT(*) FILTER (WHERE gsc_data_available IS TRUE AND gsc_impressions>0
          AND (gsc_avg_position IS NULL OR gsc_avg_position<=0)) AS unknown_position_days
        FROM daily'''
}

def make_frame(month='march'):
    """Aggregate in DuckDB; never load raw daily rows into pandas."""
    con = connection(month)
    frame = con.sql('''WITH period AS (
      SELECT client_hash_id, content_hash_id,
        CASE WHEN DAY(report_date)<=15 THEN 'past' ELSE 'future' END period,
        SUM(gsc_impressions) impressions, SUM(gsc_clicks) clicks,
        SUM(CASE WHEN gsc_avg_position>0 THEN gsc_avg_position*gsc_impressions END)
          / NULLIF(SUM(CASE WHEN gsc_avg_position>0 THEN gsc_impressions END),0) mean_position,
        STDDEV_POP(gsc_impressions)/NULLIF(AVG(gsc_impressions),0) impression_cv,
        COUNT(*) available_days,
        COUNT(*) FILTER (WHERE gsc_impressions>0) active_days,
        COUNT(*) FILTER (WHERE gsc_impressions>0 AND gsc_clicks=0) zero_click_days
      FROM daily WHERE gsc_data_available IS TRUE
      GROUP BY 1,2,3
    ) SELECT a.client_hash_id, a.content_hash_id, a.impressions past_impressions,
        a.clicks past_clicks, 100.0*a.clicks/a.impressions past_ctr,
        LN(1+a.impressions) log_impressions, a.mean_position,
        a.impression_cv,
        a.zero_click_days*1.0/NULLIF(a.active_days,0) zero_click_fraction,
        a.available_days past_available_days, b.available_days future_available_days,
        b.impressions future_impressions,
        100.0*b.clicks/NULLIF(b.impressions,0) future_ctr
      FROM period a LEFT JOIN period b
        ON a.client_hash_id=b.client_hash_id AND a.content_hash_id=b.content_hash_id AND b.period='future'
      WHERE a.period='past' AND a.impressions>=500 AND a.mean_position BETWEEN 1 AND 20
        AND a.available_days=15
      ORDER BY a.client_hash_id,a.content_hash_id''').df()
    con.close()
    assert not frame.duplicated(['client_hash_id','content_hash_id']).any()
    assert frame[FEATURES].notna().all().all()
    assert frame['past_ctr'].between(0,100).all()
    frame['position_band'] = pd.cut(frame.mean_position, [0,3,10,20], labels=['1-3','>3-10','>10-20'])
    frame['evaluation_eligible'] = ((frame.future_impressions>=500)
        & (frame.future_available_days >= (16 if month=='march' else 15))
        & frame.future_ctr.between(0,100))
    return frame

def holdout_mask(frame):
    return frame.client_hash_id.map(lambda x: int(hashlib.sha256(('ctr-v1:'+x).encode()).hexdigest()[:8],16)%5==0)

def metric_row(y, pred, frame, name):
    errors = np.abs(np.asarray(y)-np.asarray(pred))
    return {'method': name, 'pages':len(frame), 'clients':int(frame.client_hash_id.nunique()),
            'mae_pp':float(errors.mean()),
            'impression_weighted_mae_pp':float(np.average(errors,weights=frame.future_impressions)),
            'client_macro_mae_pp':float(pd.Series(errors,index=frame.index).groupby(frame.client_hash_id).mean().mean())}

def baseline_queue(frame, references=None):
    refs = frame if references is None else references
    band = refs.groupby('position_band',observed=True).past_ctr.median()
    out = frame.copy()
    out['reference_ctr'] = out.position_band.astype('string').map(band).astype(float).fillna(refs.past_ctr.median())
    out['gap_pp'] = (out.reference_ctr-out.past_ctr).clip(lower=0)
    out['score'] = out.gap_pp/100*out.past_impressions
    out['reason_code'] = 'BELOW_POSITION_REFERENCE'
    out['action'] = 'Review search intent and snippet'
    out = out.loc[out.score>0].sort_values(['score','content_hash_id'],ascending=[False,True]).reset_index(drop=True)
    out['rank'] = np.arange(1,len(out)+1)
    return out

def save_json(name, data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
