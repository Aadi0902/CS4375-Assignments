# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:36:13 2020

@author: Aadi
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Design
# K random tweets are selected as initial centroids
# assign each tweet to a centroid based on min distance 
# recalculate centroid tweet by finding the tweet with min of sum squared distance
# repeat with new centroids
# stop when the change of centroids sum of squared distances is close to 0 (pick some tolerance)

class Kmeans:
  def __init__(self,dataFile):
    df = self.preprocessData(dataFile)

  def preprocessData(self,datafile):
    df = pd.read_csv(datafile,header=None,delimiter = "|")
    df.drop(df.columns[[1,2]],axis=1,inplace=True) #removes tweedID and timestamp
    df[0].replace(to_replace='@*\b',value='',inplace=True,regex=True) # removes mentions
    df[0] = df[0].replace({'#', ''}) # remove hash symboll
    df[0].replace(to_replace='https\S+',value='',inplace=True,regex=True)#remove url
    df[0] = df[0].str.lower() #makes all letters lower case
    return df

  def jacardDistance(strList1, strList2): #
    sameWords = 0
    differentWords = 0
    for word1 in strList1:
      for word2 in strList2:
        if(word1 == word2):
          sameWords += 1
        else:
          differentWords += 1
    return 1 - differentWords / sameWords

  def distance(df):
    nElements = len(df)
    jacardDist = np.zeros((nElements,nElements))

    for ind1 in range(nElements):
      for ind2 in range(ind1, nElements):

        jacardDist[ind1][ind2] = jacardDistance(df[ind1][0].split(), df[ind2][0].split())
        jacardDist[ind2][ind1] = jacardDist[ind1][ind2]

    return jacardDist

  def clusters(k, df, centroids): # Clusters instances into k number of clusters
    n = len(df)
    #centroids = np.array(df.iat((int)(ind * n/k), 0)) for ind in range(k))
    jacardDist = self.distance(df)
    clusterId = np.zeros(n)
    bins = []
    for ind in range(k):
      bins.append(np.array())
    
    for ind1 in range(n):
      minDistance = jacardDist[centroids[0]][ind1]
      for ind2 in range(1, k):
        if (jacardDist[centroids[ind2]][ind1] < minDistance):
          minDistance = jacardDist[centroids[ind2]][ind1]
          clusterId[ind1] = ind2
      np.append( bins[clusterId[ind1]], ind1)

    return bins


  def newCentroids(k, bins): # Calculate and return the new centroids of the clusters
    centroidIds = np.zeros(k)

    for i in range(k):
      n = len(bins[k])
      bin = bins[k]
      minSSE = 0
      for ind1 in range(n):
        sse = 0
        for ind2 in range(n):
          sse += jacardDist[bin[ind1][bin[ind2]]] ** 2 #summing the square distances of points

        if ind1 == 0 or minSSE > sse:
          minSSE = sse
          centroidIDs[i] = bin[ind1]

    return centroidIDs

  def findKMeans(df, k):
    centroids = np.array((ind * n/k) for ind in range(k)) # Random centroids
    centroidsChanged = True
    while(centroidsChanged):
      bins = clusters(k, df, centroids)
      newCentroids = newCentroids(k, bins)      
      centroidsChanged = False
      for i in range(k):
        if newCentroids[i] != centroids[i]:
          centroidsChanged = True
      centroids = newCentroids