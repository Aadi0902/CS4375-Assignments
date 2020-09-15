# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 04:45:41 2020

@author: Aadi
"""
import numpy as np
import matplotlib.pyplot as plt
import preProcessing as prep

class Part1:

    def main(self):
      xTrain,yTrain,xTest,yTest = prep.processedData()  
      self.notUsingMLlibraries(xTrain,yTrain,xTest,yTest)


    def notUsingMLlibraries(self,xTrain,yTrain,xTest,yTest):
      nColumns = len(xTrain[0])
      nRows = len(xTrain)
      n = nRows
      
      # Learning rate
      alpha = [0.0001, 0.001, 0.005, 0.01,0.1]
      for i in range(len(alpha)):
      #Random initilization
        W = [[0.5]] * nColumns

        #Convert Y to column matrix
        Y = np.reshape(yTrain,(nRows,1))
        
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
        plt.plot(xAxis,yAxis,label=i)
      
      plt.legend(["Alpha = 0.0001","Alpha = 0.001","Alpha = 0.005","Alpha = 0.01","Alpha = 0.1"])
      plt.xlabel("Number of iterations")
      plt.ylabel("MSE")
      plt.show()
        #print("Calculated predict values: ",yPredict[0:10])

# Run this class
tst = Part1()
tst.main()