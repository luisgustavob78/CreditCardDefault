import pickle
import numpy as np
import pandas as pd
import joblib
from typing import List
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, conlist
import csv
from io import StringIO


app = FastAPI(title="Credit card default prediction! Upload your batch")

# Represents a batch of wines
class Batch(BaseModel):
    batches: List[List[float]]

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    contents = contents.decode("utf-8")

    # Parse the CSV content
    batch_data = Batch(csv_data=contents)
    dataframe = pd.read_csv(pd.compat.StringIO(batch_data.csv_data))

    # Load classifier from pickle file
    clf = joblib.load("../app/model.pkl")

    probs = model.predict_proba(batch_data.batches)
    
    thr = 0.55
    pred = ["default" if v > thr else "good payment" for v in probs]
    
    return {"Prediction": pred}