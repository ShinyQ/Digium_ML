from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import List
from helper import api
from io import StringIO

import pandas as pd
import numpy as np

from sklearn.neighbors import BallTree

import os

class Museum(BaseModel):
    name: str
    latitude: str
    longitude: str

class MuseumList(BaseModel):
    data: List[Museum]

app = FastAPI(debug=True)

@app.get('/', status_code=200)
def status(response: Response):
    return api.builder("Digium ML API Works!", response.status_code)


@app.post("/get_nearby_museum", status_code=200)
def get_prediction(data: MuseumList, response: Response):
    result = []

    data = """NAME Latitude Longitude
B -7.879046689728909 112.51992647162353
C -7.230805 112.7342
D -7.8890 112.5285
E -6.9014 -6.9014
"""

    df = pd.read_csv(StringIO(data), sep = ' ')

    tree = BallTree(np.deg2rad(df[['Latitude', 'Longitude']].values), metric='haversine')

    # Setup distance queries (points for which we want to find nearest neighbors)
    other_data = """NAME Latitude Longitude
B_alt -6.4014 -6.9014
"""

    df_other = pd.read_csv(StringIO(other_data), sep = ' ')

    query_lats = df_other['Latitude']
    query_lons = df_other['Longitude']

    distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k = 3)

    r_km = 6371 # multiplier to convert to km (from unit distance)
    for _, d, ind in zip(df_other['NAME'], distances, indices):
        for i, index in enumerate(ind):
            result.append([df.iloc[index]['NAME'], d[i]*r_km])
        
    return api.builder(result, response.status_code)