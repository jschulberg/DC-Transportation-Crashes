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
                                                        
    df_pivoted['PERSON'] = df_pivoted['PERSON'].str.title()
    
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

#%% Run Kmeans clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def cluster_preprocess(df): 
    '''
    Our goal here is to take our pivoted data frame and perform necessary pre-processing for clustering.
    First, we select columns of interest in four categories: impairment flags, vehicle type flags, 
    geographic columns, and columns pivoted in the previous step. We use coordinates instead of lat/long
    because clustering algorithms work better with coordinates. Because we are more interested in the 
    types of entities involved in a crash than the number of each entity, we convert the total entity
    columns into a flag.

    Parameters
    ----------
    df : DataFrame
        pivoted DataFrame.

    Returns
    -------
    df_pivoted_trim : DataFrame
        trimmed dataframe with only columns of interest, one-hot encoded where categorical

    '''
    df_pivoted_trim=df[[
        #geo columns, using coordinates instead of lat/long because clustering algos do better with coordinates.
        'XCOORD','YCOORD', 'WARD',
         #impairment cols
        'SPEEDING_INVOLVED','BICYCLISTSIMPAIRED',  'DRIVERSIMPAIRED', 'PEDESTRIANSIMPAIRED',
        #convert to vehicle flags for these cols
        'TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS','TOTAL_VEHICLES','TOTAL_TAXIS',
        #pivoted columns
        'PERSON',  'INJURY_TYPE','INJURY_COUNT']] 

    #convert total colums to a flag
    flag_cols=['TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS','TOTAL_VEHICLES','TOTAL_TAXIS']
    for i in flag_cols:
        df_pivoted_trim[i]=df_pivoted_trim.loc[df_pivoted_trim[i]>1,i]=1 #positive case
        df_pivoted_trim[i]=df_pivoted_trim.loc[df_pivoted_trim[i]<0,i]=0 #replace neg vals with 0

    #drop the crashes that occur on a ward border, usually a major street, about 0.5% of records
    df_pivoted_trim=df_pivoted_trim[df_pivoted_trim.WARD !='Null']
    df_pivoted_trim=df_pivoted_trim[df_pivoted_trim.WARD !='UNKNOWN']
    df_pivoted_trim=df_pivoted_trim.dropna(subset=['WARD'])

    #one-hot encode categorical columns
    cat_cols=['PERSON', 'INJURY_TYPE','WARD']
    for i in cat_cols:
        ohe=pd.get_dummies(df_pivoted_trim[i])
        df_pivoted_trim=df_pivoted_trim.drop(i,axis=1)
        df_pivoted_trim=df_pivoted_trim.join(ohe)
    return df_pivoted_trim.drop_duplicates()

df_pivoted_trim=cluster_preprocess(df_pivoted)

#%% Generate elbow graph
def plot_kmeans_elbow(df):
    '''
    Generate an elbow graph using preprocessed data with selected columns of interest. 
    Use to select optimal number of clusters.

    Parameters
    ----------
    df : DataFrame
        pivoted DataFrame of preprocessed data.

    Returns
    -------
    Generates an elbow graph
    '''
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(df)
        distortions.append(km.inertia_)
    # plot
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Graph for KMeans Clustering')
    plt.show()
plot_kmeans_elbow(df_pivoted_trim)


#%% Run Kmeans clustering
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def KMeans_cluster(df):
    '''
    Our goal here is to take our pivoted data frame and perform necessary pre-processing for clustering.
    First, we select columns of interest in four categories: impairment flags, vehicle type flags, 
    geographic columns, and columns pivoted in the previous step. We use coordinates instead of lat/long
    because clustering algorithms work better with coordinates. Because we are more interested in the 
    types of entities involved in a crash than the number of each entity, we convert the total entity
    columns into a flag.

    Parameters
    ----------
    df : DataFrame
        pivoted and pre-processed DataFrame.

    Returns
    -------
    df_pivoted_trim : DataFrame
        pivoted and pre-processed dataframe with cluster assignments for each n_clusters appended

    '''
    for i in range(2,10):
        km = KMeans(n_clusters=i, random_state=0)
        label=km.fit_predict(df)
        df[f'cluster {i}']=label

    df_np=df.to_numpy()

    fig, axs = plt.subplots(2,4, figsize=(16, 8), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    for ax in axs: 
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(2, 10):
        km = KMeans(n_clusters=i, random_state=0)
        label=km.fit_predict(df_np)
        u_labels = np.unique(label)
        centroids = km.cluster_centers_

        axs[i-2].set_title(f'{i} Clusters')
        for j in u_labels:
            axs[i-2].scatter(df_np[label == j , 0] , df_np[label == j , 1] ,label=j,s=2,cmap='viridis')
        axs[i-2].scatter(centroids[:,0] , centroids[:,1] , s = 30, color = 'k')
    plt.show()
    return df
KMeans_cluster(df_pivoted_trim).to_csv('Data/KMeans_cluster_results.csv')


#%% Run geospatial clustering
from sklearn.cluster import DBSCAN

def cluster_dbscan(df_pivoted, 
                   eps = .5,
                   min_samples = 10):
    '''
    Next, we'll look to see how accurate it is to define the crash epicenters
    by ward, which are pre-assigned voting districts, rather than some other
    natural method. To do so, we'll use the haversine-distance implementation
    of DBSCAN.

    Parameters
    ----------
    df_pivoted : DataFrame
        DESCRIPTION.
    eps : float, optional
        The max distance that points can be from each other to be 
        considered a cluster. The default is .1.
    min_samples : int, optional
        The minimum cluster size (everything else gets classified as noise).
        The default is 10.

    Returns
    -------
    df_clustered : DataFrame
        New dataframe with one new column: 'cluster', denoting the clusters
        that each row was assigned to.

    '''
    
    # Pull out the lat long coordinates, removing duplicates and dropping any blanks
    coordinates = df_pivoted[['LATITUDE', 'LONGITUDE']].drop_duplicates().dropna()
    
    # Convert the epsilon accordingly
    miles_per_radian = 3956
    epsilon = eps / miles_per_radian
    
    db = DBSCAN(eps = epsilon, 
                algorithm = 'ball_tree',
                metric = 'haversine',
                min_samples = 2).fit(np.radians(coordinates))
    

    # db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    cluster_labels = db.labels_
    num_clusters = len(set(cluster_labels))
    clusters = pd.Series([coordinates[cluster_labels == n] for n in range(num_clusters)])
    print('Number of clusters: {}'.format(num_clusters))
    
    # Join the clusters back in
    coordinate_clusters = pd.concat([pd.DataFrame(cluster_labels, columns = ['cluster']), 
                                     coordinates.reset_index(drop = True)], 
                                    axis = 1)
    
    df_clustered = df_pivoted.merge(coordinate_clusters,
                                    how = 'left',
                                    on = ['LATITUDE', 'LONGITUDE']
                                    )
    
    return df_clustered, cluster_labels

df_clustered, cluster_labels = cluster_dbscan(df_pivoted,
                              eps = .05,
                              min_samples = 10)
print(df_clustered['cluster'].value_counts())




#%% Compute silhouette scores
from sklearn import metrics

def compute_silhouette_score(df_clustered, cluster_labels):
    # Pull out the lat long coordinates, removing duplicates and dropping any blanks
    coordinates = df_clustered[['LATITUDE', 'LONGITUDE']].drop_duplicates().dropna()
    
    # Compute the silhouette score
    silo = metrics.silhouette_score(np.radians(coordinates), cluster_labels)
    
    return silo


# eps_vals = np.concatenate((np.arange(.01, .1, .02), np.arange(.1, .5, .1)))
# silo_scores = []
# for i in eps_vals:
#     print(f'\nRunning DBSCAN for epsilon = {i}...')
#     df_clustered_iter, cluster_labels_iter = cluster_dbscan(df_pivoted,
#                                                               eps = i,
#                                                               min_samples = 100)
    
#     print('Computing silhouette score...')
#     silo = compute_silhouette_score(df_clustered_iter, cluster_labels_iter)
    
#     print(f'Score = {silo}')
#     silo_scores.append(silo)

#%% Function to process lots of epsilon values and concatenate the results
def compute_multiple_dbscans(df_pivoted, eps_vals = []):
    # Initialize an empty dataframe to hold our results
    df_stacked = pd.DataFrame()
    
    # Loop through all of our epsilon values
    for eps in eps_vals:
        print(f'Running DBSCAN for epsilon = {eps}...')
        # Run DBSCAN
        df_clustered, cluster_labels = cluster_dbscan(df_pivoted,
                                      eps = eps,
                                      min_samples = 10)
        df_clustered['eps'] = eps
        
        # Remove any rows that aren't grouped properly
        df_clustered_filtered = df_clustered.loc[(df_clustered['cluster'] != 0), :]# | \
                                                 # (df_clustered['cluster'] != -1), :]
        
        # Concatenate the results
        df_stacked = pd.concat([df_stacked, df_clustered_filtered])
        
    return df_stacked
        
eps_vals = np.arange(.01, .1, .01)
df_stacked = compute_multiple_dbscans(df_pivoted, eps_vals = eps_vals)

print(df_stacked['eps'].value_counts())


#%% Write out our results
temp2 = df_clustered.sample(1000)
df_clustered.to_csv('Data/dc_crash_data_analyzed.csv')
df_stacked.to_csv('Data/dbscan_multiple_epsilons.csv', index = False)
