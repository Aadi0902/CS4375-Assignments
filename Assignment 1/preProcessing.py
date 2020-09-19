# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 05:06:51 2020

@author: Aadi, Andrew
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def processedData():
            
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
    header=None,delim_whitespace = True,na_values=["?"])
              
    # Drop empty rows i.e. rows with ?
    df = df.dropna()
        
    # Columns desciption:
    # mpg | cylinders | displacement | hp | weight | accn | model | origin | car_name
        
    #Separate the data into x and y variables
    x = df.iloc[:,1:7]
    y = df.iloc[:,0]
        
    #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    from sklearn.model_selection import train_test_split
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size = 0.80) # Add random_state = 3 to get consistent data similar to the report
        
    # Compute sde, mean of the data  
    scaler = StandardScaler()
    scaler.fit(xTrain)
        
    # Transform the x data
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)
              
    # Convert y data to lists
    yTrain = yTrain.tolist()
    yTest = yTest.tolist()
        
    # Concatenate a column of '1's to the x data as bias  
    xTrain = np.c_[xTrain, np.ones((len(xTrain), 1))]
    xTest = np.c_[xTest, np.ones((len(xTest), 1))]
              
    return (xTrain, yTrain, xTest, yTest)