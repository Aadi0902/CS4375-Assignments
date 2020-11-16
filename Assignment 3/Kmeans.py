# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:36:13 2020
@author: Aadi, Andrew
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Design
# K random tweets are selected as initial centroids
# assign each tweet to a centroid based on min distance 
# recalculate centroid tweet by finding the tweet with min of sum squared distance
# repeat with new centroids
# stop when the change of centroids sum of squared distances is close to 0 (pick some tolerance)

class Kmeans:

  def __init__(self,dataFile,k):#k is cluster count
    df = self.preprocessData(dataFile)
    self.SSE = 0
    self.jacardDist = np.zeros((len(df),len(df)))
    centroids = self.findKMeans(df,k)

  def preprocessData(self,datafile):
    df = pd.read_csv(datafile,header=None,delimiter = "|")
    df.drop(df.columns[[0,1]],axis=1,inplace=True) #removes tweedID and timestamp
    df.replace(to_replace='@\w+',value='',inplace=True,regex=True) # removes mentions
    df = df.replace({'#':''}, regex=True) # remove hash symbol
    df.replace(to_replace='(\s)http\S+',value='',inplace=True,regex=True)#remove url
    #df = df.str.lower() #makes all letters lower case
    df = df.apply(lambda x: x.astype(str).str.lower())
    return df

  def jacardDistance(self, strList1, strList2): #
    set1 = set(strList1)
    set2 = set(strList2)
    sameWords = set1 & set2
    totalWords = set1 | set2
    return 1 - len(sameWords) /len(totalWords)

  def makeDistanceMatrix(self, df):
    nElements = len(df)
    self.jacardDist = np.zeros((nElements,nElements))
    for ind1 in range(nElements):
      for ind2 in range(ind1, nElements):
        if ind1 == ind2:
            self.jacardDist[ind1][ind2] = 0
            self.jacardDist[ind2][ind1] = 0
            continue
        self.jacardDist[ind1][ind2] = self.jacardDistance(df[2][ind1].split(), df[2][ind2].split())
        self.jacardDist[ind2][ind1] = self.jacardDist[ind1][ind2]

  def clusters(self, k, df, centroids): # Clusters instances into k number of clusters
    n = len(df)
    self.makeDistanceMatrix(df)
    bins = []
    for ind in range(k):
      bins.append(np.array([], dtype = int))
    
    for ind1 in range(n):
      minDistance = self.jacardDist[centroids[0]][ind1]
      bestCentroid = 0
      for ind2 in range(1, k): #ind2 is cluster number (0,1,...k-1)
        if (self.jacardDist[centroids[ind2]][ind1] < minDistance):
          minDistance = self.jacardDist[centroids[ind2]][ind1]
          bestCentroid = ind2
      bins[bestCentroid] = np.append( bins[bestCentroid], ind1)

    return bins


  def newCentroids(self, k, bins): # Calculate and return the new centroids of the clusters
    centroids = np.zeros((k), int)
    SSE = 0
    for i in range(k):
      bin = bins[i]
      binSize = len(bin)
      minSSE = 0
  
      for ind1 in range(binSize): # loop 91
        #ind1 is a new potential candidate for bin i
        tempSSE = 0
        for ind2 in range(binSize): # Loop 93
          # ind2 is a new instance for distance
          tempSSE += self.jacardDist[bin[ind1]][bin[ind2]] ** 2 #summing the square distances of points
        # we have the new bin SSE for the potential new centroid for bin i
        if ind1 == 0 or minSSE > tempSSE: # check if ind1 can be the new centroid
          minSSE = tempSSE
          centroids[i] = bin[ind1]
      #minSSE is SSE of bin i
      SSE += minSSE
    self.SSE = SSE
    return centroids

  def findKMeans(self, df, k):
    centroids = np.array(random.sample(range(0, len(df)), k), dtype=int) #generate random centroids initially
    newCentroids = centroids
    centroidsChanged = True
    while(centroidsChanged):
      bins = self.clusters(k, df, centroids)
      newCentroids = self.newCentroids(k, bins)
      centroidsChanged = False
      for i in range(k):
        if newCentroids[i] != centroids[i]:
          centroidsChanged = True
      centroids = newCentroids
  
    print("Cluster 1 size: "+str(len(bins[0])))
    print("Cluster 2 size: "+str(len(bins[1])))
    print("Cluster "+str(len(bins))+" size : "+str(len(bins[len(bins)-1])))
    print("\nSSE: "+str(self.SSE))
    return centroids

if __name__ == "__main__":
  k = 10
  print("k = "+str(k))
  print("")
  kmeans = Kmeans("https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Assignment%203/Datasets/usnewshealth.txt", k)
  #kmeans = Kmeans("https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Assignment%203/Datasets/small.txt", 2)
  #small.txt used for samall dataset