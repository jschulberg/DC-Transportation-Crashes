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
import math
import requests
import json

#%%
def get_crash_records(id_url, url):
    '''
    Function to pull the crash records from the DC Open Data website. Unfortunately,
    their API only allows us to pull 1000 records at a time (and, upon testing, 
    we were only able to get it to work when we hit the API with 100 records at a time).
    To circumvent this, this function gets the full list of crash record IDs from the 
    site and then breaks them up into 100-record chunks. It then hits the API for
    each of these 100-record chunks, concatenating the results to a master DataFrame.

    Parameters
    ----------
    id_url : string
        Base URL to get the list of Object IDs for all the crash records.
    url : string
        Base URL to get the full crash record and associated information for a given
        Object ID.

    Returns
    -------
    ids : list
        List of all the Object IDs for all the crash records on the OpenData website.
    df : DataFrame
        DataFrame of all the corresponding information for all of the crash record
        Object IDs.

    '''
    
    # Start by pulling the potential IDs from the website
    response_API_id = requests.get(id_url)
    if response_API_id.status_code == 200:
    
        id_data = response_API_id.text
        
        id_json = json.loads(id_data)
        ids = id_json['objectIds']
        
        print(f'Successfully pulled {len(ids)} from the OpenData website.')
        
    # If this doesn't work, return an error message
    else: 
        print(f'Unable to pull the corresponding ids. Received an error code of {response_API_id.status_code}')
    

    # Now initialize an empty dataframe to hold all of our results
    df = pd.DataFrame()
    
    # Now loop through 1000-row chunks of our IDs, pulling the ids from the API accordingly
    for i in range(math.ceil(len(ids)/100)):
        print(f'Iteration {i}: Pulling API data for IDs {100*i}:{100*i+100}...')
        
        # Pull the 1000 necessary IDs 
        id_chunks = ids[100*i:100*i + 100]
        
        # convert it to a string so we can put it into the URL
        # and remove the beginning/end brackets so it's in the proper API format
        ids_as_string = str(id_chunks)[1:][:-1]
        new_url = url + '&objectIds=' + ids_as_string
        
        try:
            # Hit the API
            response_API = requests.get(new_url)
            data = response_API.text
            data_json = json.loads(data)
            
            # It looks like all of the data we need is located in the 'features' 
            # portion of the json
            for idx, row in enumerate(data_json.get('features')):
                # print(idx, row)
                # The data is stored in the 'attributes' key
                data_row = row.get('attributes')
                # Convert this into a temporary dataframe
                temp_df = pd.DataFrame([data_row])
                # Concatenate this to our master df
                df = pd.concat([df, temp_df])
                
        except Exception as e:
            print(f'Found the error "{e}" with iteration {i}')
    
    return ids, df

id_url = 'https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Safety_WebMercator/MapServer/24/query?where=1%3D1&outFields=*&returnIdsOnly=true&outSR=4326&f=json'
url = 'https://maps2.dcgis.dc.gov/dcgis/rest/services/DCGIS_DATA/Public_Safety_WebMercator/MapServer/24/query?where=1%3D1&outFields=*&outSR=4326&f=json'

ids, df = get_crash_records(id_url, url)





