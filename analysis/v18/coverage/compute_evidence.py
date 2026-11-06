import os
from pathlib import Path
import json,gc
import pandas as pd,numpy as np
OUT=Path(__file__).parent
ROOT=Path(os.environ["BE_DATA_ROOT"])
RUN=Path(os.environ.get("BE_RUN_DIR", ROOT/"BE_integrated_grid_v18_guarded_fixed_outputs"))
recipe=json.loads((OUT/'final_recipe.json').read_text())
cache=Path(os.environ.get("BE_FEATURE_BANK", RUN/"recovered_features"/"54f4f143685242ca"))
coverage=[];burden=[];align=[]
for cohort in ['MIMIC','eICU']:
 frame=pd.read_csv(cache/(cohort+'.csv.gz'),usecols=['patient_id','hospital_id','label','landmark','age'])
 frame=frame[frame.age.ge(18)&frame.age.notna()]
 raw=json.loads((cache/(cohort+'_raw_icd_history.json')).read_text())
 for landmark in [1,2]:
  f=frame[frame.landmark.eq(landmark)].copy()
  f['has_icd']=[raw[str(int(h))].get('total_icd_ever_hx',0)>0 for h in f.hospital_id]
  for pop in ['all_labs','icd_history']:
   pf=f if pop=='all_labs' else f[f.has_icd]
   align.append(dict(cohort=cohort,population=pop,landmark=landmark,admissions=len(pf),cases=int(pf.label.sum())))
   for label,g in pf.groupby('label'):
    rows=[raw[str(int(h))] for h in g.hospital_id]
    totals=np.array([r.get('total_icd_rate_per_adm',0) for r in rows])
    burden.append(dict(cohort=cohort,population=pop,landmark=landmark,label=int(label),n=len(g),
       fraction_any_icd=float(g.has_icd.mean()),total_rate_mean=float(totals.mean()),
       total_rate_median=float(np.median(totals)),total_rate_nonzero_median=float(np.median(totals[totals>0])) if (totals>0).any() else None))
    for cat in recipe['categories']:
     ever=np.array([r.get(cat+'_ever_hx',0) for r in rows]);rate=np.array([r.get(cat+'_rate_per_adm',0) for r in rows])
     share=np.array([r.get(cat+'_share',0) for r in rows])
     coverage.append(dict(cohort=cohort,population=pop,landmark=landmark,label=int(label),category=cat,n=len(g),
       covered=int((ever>0).sum()),coverage=float((ever>0).mean()),rate_mean=float(rate.mean()),
       rate_nonzero_median=float(np.median(rate[rate>0])) if (rate>0).any() else None,share_mean=float(share.mean())))
 del raw,frame;gc.collect()
expected=pd.read_csv(RUN/'primary'/'cohort_summary.csv')
actual=pd.DataFrame(align)
joined=expected.merge(actual,on=['cohort','population','landmark'],suffixes=('_saved','_audit'),validate='one_to_one')
assert (joined.admissions_saved==joined.admissions_audit).all() and (joined.cases_saved==joined.cases_audit).all()
pd.DataFrame(coverage).to_csv(OUT/'v18_category_coverage_verified.csv',index=False)
pd.DataFrame(burden).to_csv(OUT/'v18_coding_burden_verified.csv',index=False)
grid=pd.read_csv(RUN/'all_scenario_grid.csv')
print('STATUS',grid.groupby(['Scenario','Model','Status']).size().to_string())
print('RECONCILED',len(joined),'cohort and landmark denominators')
print(pd.DataFrame(coverage).query("population=='icd_history' and landmark==1 and label==1")[['cohort','category','covered','n','coverage']].to_string(index=False))
summary={'cohort_checks':joined.to_dict(orient='records'),'counts':grid.groupby(['Scenario','Model','Status']).size().reset_index(name='rows').to_dict(orient='records'),
 'coverage':coverage,'burden':burden,'source':'Read only aggregation of V18 pinned feature bank; no fitting or source writes'}
(OUT/'verified_coverage.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
