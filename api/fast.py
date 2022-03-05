from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import joblib
import tensorflow as tf
import tensorflow_text as text
import api.prep_encoder



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# new_model = tf.keras.models.load_model('models/sentiment_model1') #this is the local model from colab
gcp_model = tf.keras.models.load_model('models/Financial_Services_model') #this is the GCP model
finance_cluster_model = tf.keras.models.load_model('models/finance_encoder') #this is the GCP model

@app.get("/")
def index():
    return {"greeting": "Hello customer reivews"}


# @app.post("/predict")
# def upload_file(file: UploadFile = File(...)):
#     X_pred  = pd.read_csv(file.file)['review']
#     # new_model = tf.keras.models.load_model('models/sentiment_model1') better off outside so not reloaded for each call
#     prediction = new_model.predict(X_pred)
#     dict_pred = {'pred':[str(x) for x in list(prediction.reshape(-1))]}
#     print(dict_pred)
#     return dict_pred

@app.post("/predict_GCP")
def upload_file(file: UploadFile = File(...)):
    X_pred  = pd.read_csv(file.file)['review']
    prediction = gcp_model.predict(X_pred)
    dict_pred = {'pred':[float(x) for x in list(prediction.reshape(-1))]}
    print(dict_pred)
    return dict_pred



@app.post("/cluster_finance")
def upload_file(file: UploadFile = File(...)):
    X = pd.read_csv(file.file)[['review']]
    X = api.prep_encoder.pre_autoencoder(X)
    prediction= finance_cluster_model.predict(X)
    dict_cluster_x = {'X': [float(x[0]) for x in prediction]}
    dict_cluster_y = {'Y': [float(y[1]) for y in prediction]}
    dict_cluster = {'cluster_x':dict_cluster_x, 'cluster_y':dict_cluster_y}
    print(dict_cluster)
    return dict_cluster



# data_items = dict1. items()
#     data_list = list(data_items)
#     df = pd. DataFrame(data_list)
