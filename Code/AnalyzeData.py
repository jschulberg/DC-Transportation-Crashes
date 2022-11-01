#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:39:12 2022

@author: justinschulberg
"""

#%%
import pandas as pd
import numpy as np
import os

#%%
path = 'Data/dc_crash_data_cleaned.csv'
df = pd.read_csv(path)


#%%
temp = df.sample(1000)

def pivot_data(df):
    '''
    First, I'd like to pivot the data such that all of the data for injuries
    appear in just two columns. That is, we currently have 14 columns for 
    fatalities, major injuries, minor injuries, and unknown injuries for 
    cyclists, drivers, pedestrians, and passengers. I'd like to pivot this longer
    so it's just two columns:
          1. Type of Injury
          2. Description of Individual

    Parameters
    ----------
    df : DataFrame
        Unpivoted DataFrame.

    Returns
    -------
    df_pivoted : DataFrame
        Pivoted DataFrame, with more rows and less columns.

    '''

    cols_to_pivot = ['MAJORINJURIES_BICYCLIST', 'MINORINJURIES_BICYCLIST', 
                  'UNKNOWNINJURIES_BICYCLIST', 'FATAL_BICYCLIST',
                  'UNKNOWNINJURIES_DRIVER', 'FATAL_DRIVER', 'MAJORINJURIES_PEDESTRIAN',
                  'MINORINJURIES_PEDESTRIAN', 'UNKNOWNINJURIES_PEDESTRIAN',
                  'FATAL_PEDESTRIAN', 'FATALPASSENGER', 'MAJORINJURIESPASSENGER', 
                  'MINORINJURIESPASSENGER', 'UNKNOWNINJURIESPASSENGER']
    
    # Our ID Variables are any columns not mentioned above
    id_vars_ = list(set(temp.columns) - set(cols_to_pivot))
    
    df_pivoted = pd.melt(df, 
                          id_vars = id_vars_,
                         # id_vars = 'OBJECTID',
                         value_vars = cols_to_pivot,
                         var_name = 'INJURY_TYPE',
                         value_name = 'INJURY_COUNT')
    
    # The passenger columns need to include an underscore before the word 'PASSENGER'
    # so we can properly split them
    df_pivoted['INJURY_TYPE'] = df_pivoted['INJURY_TYPE'].str.replace('PASSENGER', '_PASSENGER')
    
    # Now split the INJURY_TYPE column into the type of injury and the description of the individual
    df_pivoted[['INJURY_TYPE', 'PERSON']] = df_pivoted['INJURY_TYPE'].str.split('_', expand = True)
    
    # Reformat the 'INJURY_TYPE' column
    df_pivoted['INJURY_TYPE'] = df_pivoted['INJURY_TYPE'].str.title() \
                                                        .str.replace('injuries', ' Injuries')
    
    # Unfortunately, this increases the size of our dataframe 14x; however,
    # most of the rows don't have any data in them (i.e. there are no injuries reported), 
    # but the OBJECTID is reported 14x. Let's replace any of the rows where this happens
    # with just 1 row
    df_grouped = df_pivoted.groupby('OBJECTID') \
                            .sum('INJURY_COUNT') \
                            .reset_index() \
                            .rename(columns = {'INJURY_COUNT': 'TOTAL_INJURIES'})
    df_grouped = df_grouped[['OBJECTID', 'TOTAL_INJURIES']]                    
        
    # Now merge this back into our original dataframe
    df_merged = pd.merge(df_pivoted,
                         df_grouped,
                         how = 'left',
                         on = 'OBJECTID')
    
    # Now if any OBJECTID's have 0 total injuries, let's rename all the injuries
    # as 'Total Injuries' so we can drop_duplicates() properly
    df_merged.loc[df_merged['TOTAL_INJURIES'] == 0, 'INJURY_TYPE'] = 'Total Injuries'
    
    # Let's also remove any rows where we have multiple types of injuries reported,
    # but the rest of the rows for that OBJECTID are 0
    df_merged = df_merged.loc[(df_merged['INJURY_COUNT'] > 0) | (df_merged['INJURY_TYPE'] == 'Total Injuries')]
    
    return df_merged.drop_duplicates(['OBJECTID', 'INJURY_TYPE', 'INJURY_COUNT'])

df_pivoted = pivot_data(df)

print(df_pivoted['INJURY_TYPE'].value_counts())
print('\n', df_pivoted['PERSON'].value_counts())

#%% Write out our results
df_pivoted.to_csv('Data/dc_crash_data_analyzed.csv')