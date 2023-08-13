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
import json


app = FastAPI(title="Credit card default prediction! Upload your json batch")

# # Represents a batch of wines
# class Credit(BaseModel):
#     batches: List[conlist(item_type=float, min_items=24, max_items=24)]

# @app.post("/upload_json/")
# async def upload_json(file: UploadFile = File(...)):
#     global json_file
#     contents = await file.read()
#     json_file = contents.decode("utf-8")

@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    global clf
    clf = joblib.load("model.pkl")

@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. This new version allows for batching. Now head over to http://localhost:81/docs"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    contents = contents.decode("utf-8")

    # Parse JSON content
    json_data = json.loads(contents)

    np_batches = np.array(json_data["batches"], dtype=float)

    names = pd.read_csv("col_names.csv")
    col_names = names["col_names"].values
    df_batch = pd.DataFrame(np_batches)
    df_batch.columns = col_names

    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    def negative_cat(value):
        if value < 0:
            value = value*(-15)
        
        else:
            pass
    
        return value

    for c in cat_cols:
        df_batch[c] = df_batch[c].apply(negative_cat)
    
    probs = clf.predict_proba(df_batch)
    probs = [p[1] for p in probs]
    thr = 0.55
    pred = ["default" if v > thr else "good payment" for v in probs]
    
    return {"Prediction": pred}