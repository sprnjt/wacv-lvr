import pandas as pd, numpy as np, sys, json
R="hf_release/culturehaystack-results/models"
M=['gemma4','qwen3vl','qwen25vl','gemma3_4b','granite4v','lfm25v','gemma3_12b']
def load(m,sub):
    f={'main':f'{R}/{m}/main_study/scored_{m}.csv','main12':f'{R}/{m}/main_study_160/scored_{m}.csv','full':f'{R}/{m}/full_pool/scored_{m}.csv'}[sub]
    try: return pd.read_csv(f)
    except FileNotFoundError: return None
def per_target(d,mode,col):
    d=d[(d['mode']==mode)&(((d.exp=='E1')&(d.N<=4))|((d.exp=='E2')&(d.N==5)))]
    return d.groupby(['target','N'])[col].mean().unstack('N')  # targets x N
def stats(P,K=None):
    a=P.mean(0).values
    acc1,acc5=a[0],a[4]; drop=a[0]-a[1]; slope=np.polyfit([2,3,4,5],a[1:5],1)[0]
    known=P[1]==1; ret=P.loc[known,5].mean()
    return np.array([acc1,acc5,acc1-acc5,drop,slope,ret])
def boot(P,B=2000,seed=0):
    rng=np.random.default_rng(seed); X=P.values; n=len(X); out=[]
    for _ in range(B):
        S=pd.DataFrame(X[rng.integers(0,n,n)],columns=P.columns); out.append(stats(S))
    out=np.array(out); return np.nanpercentile(out,2.5,0),np.nanpercentile(out,97.5,0)
res={}
for sub in ['main','full']:
    for m in M:
        d=load(m,'main12' if (m=='gemma3_12b' and sub=='main') else sub)
        if d is None: continue
        r={}
        P=per_target(d,'mc','joint_ok'); r['n']=len(P); r['est']=stats(P).tolist(); lo,hi=boot(P); r['lo']=lo.tolist(); r['hi']=hi.tolist()
        for c in ['item_ok','country_ok']:
            Q=per_target(d,'mc',c); r[c]=[Q[1].mean(),Q[5].mean()]
        O=per_target(d,'open','joint_ok'); r['open1']=float(O[1].mean()) if len(O) else None
        res[f'{sub}:{m}']=r; print(sub,m,r['n'],np.round(r['est'],3),r['open1'],file=sys.stderr)
json.dump(res,open('work/t2/res.json','w'))
