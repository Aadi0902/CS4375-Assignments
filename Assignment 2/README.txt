README

Authors:
Aadi Kothari and Andrew Su

Directions:
1. Open NeuralNet.py
2. Change activationFunc variable in line 255 to switch activation functions (sigmoid, tanh, relu)
3. Run NeuralNet.py

Files:
1. NeuralNet.py - This file contains the entire project, including the preprocesing, 
    activation functions, and the back propogation algorithm.
2. assignment2_Report.pdf - Theoretical part of the assignment 

Libraries:
numpy (numpy.org)
pandas (pandas.pydata.org)
sklearn (scikit-learn.org)


Dataset:
Wall-Following Robot Navigation Data Data Set

Dataset link:
https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_4.data

This data set contained distance readings from various sensors on a robot. Each tuple is classified as 
one of four different directions the robot would move: Move-Forward, Slight-Right-Turn, Sharp-Right-Turn,
or Slight-Left-Turn. Our implementation of a Neural Netowork predicts the move the robot makes using
the sensor data.  

Attributes:
1. Front sensor
2. Left sensor
3. Right sensor
4. Rear sensor
Classes: Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, and Slight-Left-Turn

Multiple data sets are given varying in number of sensors' data used.  We chose the dataset
that used 4 sensors. Clases are mapped to one-hot vectors, [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]
in order to do binary classification.

Additional Dataset information:
https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data