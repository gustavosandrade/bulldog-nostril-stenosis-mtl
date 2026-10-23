"""Recompute real archived summaries without external packages or training."""
from pathlib import Path
import csv,json,math,statistics,itertools
R=Path(__file__).resolve().parent
labels=['mild','moderate','severe']
read=lambda name:list(csv.DictReader((R/name).open()))
preds=read('predictions/oof_predictions.csv');folds=read('predictions/fold_metrics.csv')
assign=read('manifests/fold_assignments.csv')
test={r['sample_id']:(int(r['fold']),r['group_id']) for r in assign if r['split']=='real_test'}
assert len(test)==190 and len({g for _,g in test.values()})==95
for f in range(1,11):
    groups={s:{r['group_id'] for r in assign if int(r['fold'])==f and r['split']==s} for s in ['internal_train','internal_validation','real_test']}
    assert all(not groups[a]&groups[b] for a,b in itertools.combinations(groups,2))
def wilcoxon(ds):
    ds=[v for v in ds if v!=0];ranks=[]
    vals=sorted(abs(v) for v in ds)
    for d in ds:
        ranks.append(statistics.mean(i+1 for i,v in enumerate(vals) if v==abs(d)))
    obs=sum(r for r,d in zip(ranks,ds) if d>0);threshold=min(obs,sum(ranks)-obs)
    return sum(min(w,sum(ranks)-w)<=threshold+1e-12 for w in (sum(r*s for r,s in zip(ranks,bits)) for bits in itertools.product([0,1],repeat=len(ds))))/(2**len(ds))
report={}
for backbone in ['mobilenetv3','resnet18']:
    names=sorted({r['condition'] for r in preds if r['backbone']==backbone});assert len(names)==6
    scores={name:{r['fold']:float(r['macro_f1_archived']) for r in folds if r['backbone']==backbone and r['condition']==name} for name in names}
    out={};pvalues={}
    for name in names:
        rows=[r for r in preds if r['backbone']==backbone and r['condition']==name];assert len(rows)==190
        assert len({r['sample_id'] for r in rows})==190
        cm=[[0]*3 for _ in range(3)]
        for r in rows:
            assert test[r['sample_id']]==(int(r['fold'].split('_')[-1]),r['group_id'])
            assert math.isclose(sum(float(r['prob_'+x]) for x in labels),1,abs_tol=1e-5)
            cm[labels.index(r['true_class'])][labels.index(r['predicted_class'])]+=1
        assert [sum(row) for row in cm]==[59,66,65]
        f1=[2*cm[i][i]/(sum(cm[i])+sum(cm[j][i] for j in range(3))) for i in range(3)]
        out[name]={'fold_mean':statistics.mean(scores[name].values()),'fold_sd':statistics.stdev(scores[name].values()),
                   'pooled_macro_f1':statistics.mean(f1),'class_f1':dict(zip(labels,f1)),'cm':cm,
                   'ordinal_mae':sum(abs(i-j)*cm[i][j] for i in range(3) for j in range(3))/190,
                   'extreme_error_rate':(cm[0][2]+cm[2][0])/190}
        if name!='baseline_categorical':
            ds=[scores[name][f]-scores['baseline_categorical'][f] for f in sorted(scores[name])]
            out[name]['delta']=statistics.mean(ds);out[name]['positive_folds']=sum(d>0 for d in ds)
            pvalues[name]=wilcoxon(ds);out[name]['p_raw']=pvalues[name]
    highest=0
    for rank,(name,p) in enumerate(sorted(pvalues.items(),key=lambda x:x[1])):
        highest=max(highest,min(1,(5-rank)*p));out[name]['p_holm']=highest
    report[backbone]=out
main=report['resnet18']['mtl_categorical_corrigido_lambda2']
assert abs(main['p_holm']-0.048828125)<1e-12
assert f"{main['delta']:.3f}"=='0.078'
(R/'derived_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'conditions':12,'predictions':len(preds),'status':'passed','primary_contrast':main},indent=2))
