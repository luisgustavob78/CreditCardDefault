import pickle
import numpy as np
import pandas as pd
import FeatureGenerator as fg
import joblib
from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, conlist
import csv
from io import StringIO


app = FastAPI(title="Credit card default prediction! Upload your json batch")

# Represents a batch of wines
class Credit(BaseModel):
    batches: List[conlist(item_type=float, min_items=24, max_items=24)]


@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    clf = joblib.load("../app/model.pkl")

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. This new version allows for batching. Now head over to http://localhost:81/docs"


@app.post("/predict")
def predict(credit: Credit):
    batches = credit.batches
    np_batches = np.array(batches)
    names = pd.read_csv("../inputs/col_names.csv")
    col_names = names["col_names"].values
    df_batch = pd.DataFrame(np_batches)
    df_batch.columns = col_names

    probs = clf.predict_proba(df_batch.batches)
    
    thr = 0.55
    pred = ["default" if v > thr else "good payment" for v in probs]
    
    return {"Prediction": pred}