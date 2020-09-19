# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 05:25:59 2020

@author: Aadi, Andrew
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
      
      print("Part 2: Using ML libraries")
      print("First 10 values:\n")
      print("Predicted MPG values    Actual MPG values")
      for ind in range(10):
        print("   %f\t\t    %f" % (yPredict[ind],yTest[ind]))
        
#      Used for MSE calculations  
#      E = yTest - yPredict
#      MSE = (1/(2*len(xTest)))*(np.dot(E.T,E))
#      print(MSE)
tst = Part2()
tst.main()