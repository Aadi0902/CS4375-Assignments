README
Authors: Aadi Kothari and Andrew Su


Directions:
1. Open NeuralNet.py
2. Change activationFunc variable in line 249 to switch activation functions (sigmoid, tanh, relu). Sigmoid is default.
3. Change max_iterations or learning_rate on line 250. 5000 and 0.001 are defaults.
4. Run NeuralNet.py

Files:
1. NeuralNet.py - This file contains the entire project, including the preprocessing, activation functions, and the back propagation algorithm.
2. assignment2_Report.pdf - Report summarizing our results.
3. assignment2_part1.pdf - Theoretical Part.

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
or Slight-Left-Turn. Our implementation of a Neural Network predicts the move the robot makes using
the sensor data.  

Attributes:
1. Front sensor
2. Left sensor
3. Right sensor
4. Rear sensor
Classes: Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, and Slight-Left-Turn

Multiple data sets are given varying in number of sensors' data used.  We chose the dataset
that used 4 sensors. Classes are mapped to one-hot vectors, [0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]
in order to do binary classification.

Additional Dataset information:
https://archive.ics.uci.edu/ml/datasets/Wall-Following+Robot+Navigation+Data

Train/Test Ratio: 0.80/0.20, random_state=3
