from pathlib import Path
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.api import app
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def client():
    with TestClient(app) as c:yield c

def test_splits_disjoint():
    d=np.load(ROOT/'reports/splits.npz');groups=[set(d[k]) for k in ['train','validation','test']]
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert len(set.union(*groups))==12000

def test_metrics():
    from sklearn.metrics import recall_score,average_precision_score
    m=json.loads((ROOT/'reports/metrics.json').read_text());d=np.load(ROOT/'reports/test_predictions.npz')
    assert recall_score(d['y'],d['probability']>=m['threshold'])==pytest.approx(m['test_selected']['recall'])
    assert average_precision_score(d['y'],d['probability'])==pytest.approx(m['test_selected']['average_precision'])
    assert m['test_always_negative']['accuracy']==.95 and m['test_always_negative']['recall']==0

def test_http(client):
    assert client.get('/health').json()['status']=='ok'
    assert client.get('/').status_code==200
    assert client.get('/docs').status_code==200
    example=client.get('/example').json();a=client.post('/predict',json=example)
    assert a.status_code==200 and 0<=a.json()['positive_score']<=1
    assert a.json()==client.post('/predict',json=example).json()

@pytest.mark.parametrize('data',[{}, {'features':[0]*11},{'features':[0]*13},{'features':[31]*12},{'features':[0]*12,'label':1}])
def test_reject_bad_input(client,data):assert client.post('/predict',json=data).status_code==422

def test_parity(client):
    import joblib
    d=client.get('/example').json();m=joblib.load(ROOT/'artifacts/model.joblib')
    p=m['model'].predict_proba([d['features']])[0,1]
    assert client.post('/predict',json=d).json()['positive_score']==pytest.approx(p)
