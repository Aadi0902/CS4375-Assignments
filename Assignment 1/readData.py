# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 10:38:28 2020

@author: Aadi Kothari
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import SGDClassifier

class LinearRegression:

    def main(self):
    
      df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                      header=None,delim_whitespace = True,na_values=["?"])
      # Drop empty rows
      df = df.dropna()

      # Columns desciption:
      # mpg | cylinders | displacement | hp | weight | accn | model | origin | car_name

      #Remove data with hp = '?'
      #df = df[df[3]!='?']

      # Drop redundant values
      #df.drop_duplicates(keep='first',inplace=True)

      # Number of rows and columns
      nRows = len(df)
      nColumns = len(df.columns)

      # Define percentage of data for training 
      train_percent = 0.8

      #Training dataframe
      dfTrain = df[0 : int(train_percent*nRows)]

      #Testing dataframe
      dfTest = df[int(train_percent*nRows):nRows]

      #Separate the data into x and y variables
      x = df.iloc[:,1:7]
      y = df.iloc[:,0]

      #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
      from sklearn.model_selection import train_test_split
      xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size = 0.80)

      scaler = StandardScaler()
      scaler.fit(xTrain)

      # Computes mean and sde and then transforms it
      xTrain = scaler.transform(xTrain)
      xTest = scaler.transform(xTest)
      yTrain = yTrain.tolist()
      yTest = yTest.tolist()

      xTrain = np.c_[xTrain, np.ones((len(xTrain), 1))]
      xTest = np.c_[xTest, np.ones((len(xTest), 1))]

      print("Actual Y test values: ",yTest[0:10])
      self.UsingMLlibraries(xTrain,yTrain,xTest,yTest)
      self.notUsingMLlibraries(xTrain,yTrain,xTest,yTest)


    def notUsingMLlibraries(self,xTrain,yTrain,xTest,yTest):
      nColumns = len(xTrain[0])
      nRows = len(xTrain)
      n = nRows

      for i in range(5):
      #Random initilization
        W = [[0.5]] * nColumns

        #Convert Y to column matrix
        Y = np.reshape(yTrain,(nRows,1))

        # Learning rate
        alpha = [0.0001, 0.001, 0.01, 0.1, 0.2]
        plt.legend(["Alpha = 0.0001","Alpha = 0.001","Alpha = 0.01","Alpha = 0.1","Alpha = 0.2"])
        xAxis = [1]*100
        yAxis = [1]*100
        for ind in range(50000):
          H = np.dot(xTrain,W)

          # Error
          E = H - Y

          # Mean square error
          MSE = (1/(2*n))*(np.dot(E.T,E))

          # Partial derivative of MSE wrt W
          pd_MSE = (1/n)*((E*xTrain).sum(axis=0)) #https://stackoverflow.com/questions/51624235/python-getting-dot-product-for-two-multidimensional-arrays
          pd_MSE = np.reshape(pd_MSE,(nColumns,1))

          W = W - alpha[i]*pd_MSE

          if(ind%500==0):
            xAxis[int(ind/500)] = ind
            yAxis[int(ind/500)] = MSE[0]
        yPredict = np.dot(xTest,W)
        plt.plot(xAxis,yAxis)
        print("Calculated predict values: ",yPredict[0:10])
      #print(E.T)
    def UsingMLlibraries(self,xTrain,yTrain,xTest,yTest):

      sgdR = SGDRegressor(tol=1e-4, penalty=None)
      sgdR.n_iter = np.ceil(10000000)
      sgdR.fit(xTrain,yTrain)

      yPredict = sgdR.predict(xTest)
      print("ML library predicted values: ",yPredict[0:10])

tst = LinearRegression()
tst.main()