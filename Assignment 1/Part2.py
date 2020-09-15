# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 05:25:59 2020

@author: Aadi
"""
from sklearn.linear_model import SGDRegressor
import preProcessing as prep
import numpy as np

class Part2:
    def main(self):
        xTrain,yTrain,xTest,yTest = prep.processedData()
        self.UsingMLlibraries(xTrain,yTrain,xTest,yTest)
    def UsingMLlibraries(self,xTrain,yTrain,xTest,yTest):

      sgdR = SGDRegressor(tol=1e-4, penalty=None)
      sgdR.n_iter = np.ceil(10000000)
      sgdR.fit(xTrain,yTrain)

      yPredict = sgdR.predict(xTest)
      print("ML library predicted values: ",yPredict[0:10])

tst = Part2()
tst.main()