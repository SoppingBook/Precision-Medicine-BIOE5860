import os
from pathlib import Path
import json,gc
import pandas as pd,numpy as np
BASE=Path(__file__).parent
ROOT=Path(os.environ["BE_DATA_ROOT"])
RUN=Path(os.environ.get("BE_RUN_DIR", ROOT/"BE_integrated_grid_v18_guarded_fixed_outputs"))
CACHE=Path(os.environ.get("BE_FEATURE_BANK", RUN/"recovered_features"/"54f4f143685242ca"))
audit=pd.read_csv(RUN/'primary'/'unified_grid_fit_audit.csv')
scope='onehot+rates+shares+ever+icd_ever+lab_current+lab_history'
anchors=audit[audit.population.eq('icd_history')&audit.fold.eq(-1)&audit.scope.eq(scope)&audit.sampling.eq('none')]
assert len(anchors)==2
def family(n):
 if n.startswith('T-'):return 'onehot'
 if n.startswith('ICD_'):return 'icd_ever'
 if n.endswith('_rate_per_adm'):return 'rates'
 if n.endswith('_share'):return 'shares'
 return 'ever'
rows=[]
for cohort in ['MIMIC','eICU']:
 frame=pd.read_csv(CACHE/(cohort+'.csv.gz'),usecols=['hospital_id','label','landmark','age'])
 frame=frame[frame.age.ge(18)&frame.age.notna()]
 raw=json.loads((CACHE/(cohort+'_raw_icd_history.json')).read_text())
 frame=frame[[raw[str(int(h))].get('total_icd_ever_hx',0)>0 for h in frame.hospital_id]]
 for anchor in anchors.itertuples():
  f=frame[frame.landmark.eq(anchor.landmark)]
  for n in anchor.selected_ICD_features.split('|'):
   v=np.array([raw[str(int(h))].get(n,0) for h in f.hospital_id],float)
   cases=f.label.to_numpy()==1
   rows.append(dict(cohort=cohort,landmark=int(anchor.landmark),feature=n,family=family(n),admissions=len(f),
    nonzero=int((v!=0).sum()),prevalence=float((v!=0).mean()),case_prevalence=float((v[cases]!=0).mean()),
    control_prevalence=float((v[~cases]!=0).mean()),mean=float(v.mean()),sd=float(v.std()),
    nonzero_median=float(np.median(v[v!=0])) if (v!=0).any() else None))
 del raw,frame;gc.collect()
d=pd.DataFrame(rows);d.to_csv(BASE/'v18_selected_feature_portability.csv',index=False)
summary=d.assign(active=d.nonzero.gt(0),constant=d.sd.eq(0)).groupby(['cohort','landmark','family']).agg(
 selected_features=('feature','size'),active_features=('active','sum'),constant_features=('constant','sum')).reset_index()
summary.to_csv(BASE/'v18_selected_feature_portability_summary.csv',index=False)
print(summary.to_string(index=False))
