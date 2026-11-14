from pathlib import Path
from contextlib import asynccontextmanager
import json, joblib
from typing import Annotated
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
ROOT=Path(__file__).resolve().parents[1]
state={}
@asynccontextmanager
async def lifespan(app):
    state['bundle']=joblib.load(ROOT/'artifacts/model.joblib');yield;state.clear()
app=FastAPI(title='Rare Event Lab',lifespan=lifespan)
class Input(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    features:list[Annotated[float,Field(ge=-30,le=30)]]=Field(min_length=12,max_length=12)
@app.get('/')
def ui():return FileResponse(ROOT/'static/index.html')
@app.get('/health')
def health():return {'status':'ok'}
@app.get('/metrics')
def metrics():return json.loads((ROOT/'reports/metrics.json').read_text())
@app.get('/example')
def example():return json.loads((ROOT/'artifacts/example.json').read_text())
@app.post('/predict')
def predict(d:Input):
    b=state['bundle'];p=float(b['model'].predict_proba([d.features])[0,1]);t=b['threshold']
    return {'positive_score':p,'threshold':t,'rare_event_flag':bool(p>=t),
        'warning':'Synthetic features only. Score is not calibrated or suitable for real people or financial decisions.'}
