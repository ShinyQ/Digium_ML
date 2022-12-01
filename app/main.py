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
    latitude: float
    longitude: float

class MuseumList(BaseModel):
    items: List[Museum]
    others: Museum

app = FastAPI(debug=True)

@app.get('/', status_code=200)
def status(response: Response):
    return api.builder("Digium ML API Works!", response.status_code)


@app.post("/get_nearby_museum", status_code=200)
def get_prediction(list: MuseumList, response: Response):
    item_list = list.items       
    museum_list = []

    for data in item_list:
        museum_list.append([data.name, data.latitude, data.longitude])

    result = []

    df = pd.DataFrame(museum_list, columns =['NAME', 'Latitude', 'Longitude'])

    tree = BallTree(np.deg2rad(df[['Latitude', 'Longitude']].values), metric='haversine')

    others = [[
        list.others.name, 
        list.others.latitude, 
        list.others.longitude
    ]]

    df_other =  pd.DataFrame(others, columns =['NAME', 'Latitude', 'Longitude'])

    query_lats = df_other['Latitude']
    query_lons = df_other['Longitude']

    distances, indices = tree.query(np.deg2rad(np.c_[query_lats, query_lons]), k = 3)

    r_km = 6371

    for _, d, ind in zip(df_other['NAME'], distances, indices):
        for i, index in enumerate(ind):
            museum = dict(item_list[index])
            museum['distance'] = round(d[i]*r_km, 0)
            result.append(museum)
        
    return api.builder(result, response.status_code)