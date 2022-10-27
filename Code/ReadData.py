#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 08:37:27 2022

@author: A bunch of us

Script to analyze the DC crashes dataset.

"""

#%%
import pandas as pd
import numpy as np
import os
import rarfile
import requests
import json


#%%
path = 'Data/'

print(os.listdir(path))

df = pd.read_csv(path + os.listdir(path)[1])


#%%
rar_path = rarfile.RarFile(path + os.listdir(path)[6])
rarfile.RarFile.open(rar_path, 'test')

#%%
url = 'https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Safety_WebMercator/MapServer/24/query?where=1%3D1&outFields=*&outSR=4326&f=json'
response_API = requests.get(url)
print(response_API.status_code)

data = response_API.text

data_json = json.loads(data)

# It looks like all of the data we need is located in the 'features' 
# portion of the json
for idx, row in enumerate(data_json.get('features')):
    print(idx, row)
    
    # The data is saved in dictionaries in both 'attributes' and 'geometry'









