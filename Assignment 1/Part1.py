# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 04:45:41 2020

@author: Aadi, Andrew
"""
import numpy as np
import matplotlib.pyplot as plt
import preProcessing as prep
from sklearn.metrics import r2_score

class Part1:

    def main(self):
      xTrain,yTrain,xTest,yTest = prep.processedData()  
      self.notUsingMLlibraries(xTrain,yTrain,xTest,yTest)


    def notUsingMLlibraries(self,xTrain,yTrain,xTest,yTest):
      nColumns = len(xTrain[0])
      nRows = len(xTrain)
      n = nRows

      
      # Optimal learning rate - see report for details
      alpha = 0.1
      
      #Random initilization
      W = [[0.5]] * nColumns

      #Convert Y to column matrix
      Y = np.reshape(yTrain,(nRows,1))
      
#      Used for plotting graphs and calculating MSE
#      Y_test = np.reshape(yTest,(len(yTest),1))
#      xAxis = [1]*250
#      yAxis = [1]*250
      
      
      for ind in range(460): # Optimal number of iterations - see report
        H = np.dot(xTrain,W)

        # Error
        E = H - Y
        
        # Mean square error
        # MSE = (1/(2*n))*(np.dot(E.T,E))

        # Partial derivative of MSE wrt W
        pd_MSE = (1/n)*((E*xTrain).sum(axis=0)) #https://stackoverflow.com/questions/51624235/python-getting-dot-product-for-two-multidimensional-arrays
        pd_MSE = np.reshape(pd_MSE,(nColumns,1))

        W = W - alpha*pd_MSE
        
#       Following code was used for MSE analysis and plots:
#        if(ind%4==0):
#          H_test = np.dot(xTest,W)
#          E_test = H_test - Y_test
#          n_test = len(xTest)
#          MSE_test = (1/(2*n_test))*(np.dot(E_test.T,E_test))
#          
#          xAxis[int(ind/4)] = ind
#          yAxis[int(ind/4)] = MSE_test[0]
#      print(yAxis[249])
#      r2 = r2_score(yTest, yPredict)
#      print(f'r2: {r2}')  
        
      # Calculate the predicted values  
      yPredict = np.dot(xTest,W)
      
      print("Part 1: Using Gradient Descent")
      print("First 10 values:\n")
      
      print("Predicted MPG values    Actual MPG values")
      for ind in range(10):
        print("   %f\t\t    %f" % (yPredict[ind],yTest[ind]))
        
#     Used for different learning rate plots:        
#     plt.legend(["Alpha = 0.00001","Alpha = 0.0001","Alpha = 0.001","Alpha = 0.005","Alpha = 0.1"])
               
#     plt.plot(xAxis,yAxis)
        
#     Used for finding optimal iterations before overfitting:        
#     plt.plot(xAxis[220:240],yAxis[220:240])
#     plt.xlabel("Number of iterations")
#     plt.ylabel("MSE")
#     plt.show()

# Run this class
tst = Part1()
tst.main()