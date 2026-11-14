"""Synthetic 95/5 rare-event lab: no real financial or customer records."""
from pathlib import Path
import json, hashlib
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, fbeta_score, accuracy_score, confusion_matrix, precision_recall_curve)
ROOT=Path(__file__).resolve().parents[1]
SEED=42

def report(y,p,threshold):
    label=(p>=threshold).astype(int)
    return {'average_precision':float(average_precision_score(y,p)),
        'roc_auc':float(roc_auc_score(y,p)), 'precision':float(precision_score(y,label,zero_division=0)),
        'recall':float(recall_score(y,label,zero_division=0)),
        'f2':float(fbeta_score(y,label,beta=2,zero_division=0)),
        'accuracy':float(accuracy_score(y,label)), 'confusion_matrix':confusion_matrix(y,label).tolist()}

def main():
    X,y=make_classification(n_samples=12000,n_features=12,n_informative=6,n_redundant=3,
        weights=[.95,.05],flip_y=0,class_sep=1.0,random_state=SEED)
    ids=np.arange(len(y));trainval,test=train_test_split(ids,test_size=.2,stratify=y,random_state=SEED)
    train,val=train_test_split(trainval,test_size=.25,stratify=y[trainval],random_state=SEED)
    candidates={'logistic':LogisticRegression(max_iter=1000,random_state=SEED),
        'balanced_logistic':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=SEED),
        'gradient_boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,
            l2_regularization=1,random_state=SEED)}
    fitted={};validation={}
    for name,est in candidates.items():
        model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),est)
        model.fit(X[train],y[train]);p=model.predict_proba(X[val])[:,1]
        fitted[name]=model;validation[name]=float(average_precision_score(y[val],p))
    selected=max(validation,key=validation.get);model=fitted[selected]
    vp=model.predict_proba(X[val])[:,1]
    thresholds=np.linspace(.01,.99,99)
    # Threshold selection uses validation ONLY, maximizing minority-sensitive F2.
    threshold=float(max(thresholds,key=lambda t: fbeta_score(y[val],vp>=t,beta=2,zero_division=0)))
    tp=model.predict_proba(X[test])[:,1]
    result={'seed':SEED,'data':'Entirely synthetic make_classification, 12000 rows,12 features,95/5,flip_y=0.',
        'split':{n:{'rows':len(ix),'positive':int(y[ix].sum())} for n,ix in [('train',train),('validation',val),('test',test)]},
        'data_sha256':hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),
        'validation_average_precision':validation,'selected_model':selected,'threshold':threshold,
        'threshold_rule':'Max validation F2 over 0.01..0.99; model selected by validation AP. No test tuning.',
        'test_selected':report(y[test],tp,threshold),'test_default_threshold':report(y[test],tp,.5),
        'test_always_negative':report(y[test],np.zeros(len(test)),.5),
        'limitations':'Synthetic IID benchmark only. Not real fraud detection or a calibrated operational risk score.'}
    (ROOT/'reports/metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    joblib.dump({'model':model,'threshold':threshold},ROOT/'artifacts/model.joblib')
    np.savez_compressed(ROOT/'reports/splits.npz',train=train,validation=val,test=test)
    np.savez_compressed(ROOT/'reports/test_predictions.npz',y=y[test],probability=tp)
    (ROOT/'artifacts/example.json').write_text(json.dumps({'features':X[test[0]].tolist()}))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    precision,recall,_=precision_recall_curve(y[test],tp)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(recall,precision,label=f"Selected model AP={result['test_selected']['average_precision']:.3f}",color='#18786c')
    ax.axhline(y[test].mean(),ls='--',label='Positive prevalence: 0.05',color='#9a5d2a')
    ax.set(xlabel='Recall',ylabel='Precision',title='Synthetic rare-event test set',xlim=(0,1),ylim=(0,1.02));ax.legend();fig.tight_layout();fig.savefig(ROOT/'reports/precision_recall.png',dpi=150)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
