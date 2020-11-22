#####################################################################################################################
#   Project: Neural Network Programming
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h             - number of neurons in the hidden layer
#   X             - vector of features for trainging instances
#   xTest         - vector of features for testing instances
#   y             - output for each training instance
#   yTest         - output for each testing instance in integer array form
#   yTestString   - ouptput for each testing instance in String form (Type of motion)
#   yPredict      - output for each predicted instacne in integer array form
#   yPredctString - output for each predicted instance in String form (Type of motion)
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#
#####################################################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNet:
    def __init__(self, dataFile, header=True, h=[5,5]):  #values of h array correspond to nodes in the layer
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        self.nL = len(h)
        self.X, self.xTest, self.y, self.yTest = self.preprocessDataAssign(dataFile)

        # Find number of input and output layers from the dataset
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        ##self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        ##self.Wb_hidden = 2 * np.random.random((1, h)) - 1
        # Create a 3D array here for inner layers
        # Create list of W_hidden and Wb_hidden, 1 for each hidden layer
        self.W_hidden = []
        self.Wb_hidden = []
        #first hidden layer weights
        self.W_hidden.append(2 * np.random.random((input_layer_size, h[0])))
        self.Wb_hidden.append(2 * np.random.random((1, h[0])))

        for i in range(1, self.nL):
            self.W_hidden.append(2 * np.random.random((h[i - 1], h[i])))
            self.Wb_hidden.append(2 * np.random.random((1, h[i])))

        self.W_output = 2 * np.random.random((h[self.nL - 1], self.output_layer_size))
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        ##self.deltaHidden = np.zeros((h, 1))
        self.deltaHidden = []
        for num_nodes in h:
            self.deltaHidden.append(np.zeros((num_nodes, 1)))

        self.h = h

    # Define activation functions
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)
        if activation == "relu":
            self.__relu(self, x)

    # Define derivative of activation functions
    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)

    # Define individual activation functions
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))

    def __relu(self, x):
        return np.maximum(0, x)

    # Define individual deivatives
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return (1 - x**2)

    def __relu_derivative(self, x):
        return np.heaviside(x, 0.0)

    # Below is the training function

    def train(self, activation="sigmoid", max_iterations=5000, learning_rate=0.001):
        # learning rate: sigmoid -> 0.001, tanh->0.0001, relu-> 0.00001
        for iteration in range(max_iterations):
            out = self.forward_pass(self.X, activation)
            error = 0.5 * np.power(out-self.y, 2)

            self.backward_pass(out, activation)
            
            #W_output is caluclated using last X_hidden
            self.W_output += learning_rate * np.dot(self.X_hidden[self.nL - 1].T, self.deltaOut)
            self.Wb_output += learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)

            for i in range(1, self.nL):
              self.W_hidden[i] += learning_rate * np.dot(self.X_hidden[i - 1].T, self.deltaHidden[i])
              self.Wb_hidden[i] += learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden[i])
            
            self.W_hidden[0] += learning_rate * np.dot(self.X.T, self.deltaHidden[0])
            self.Wb_hidden[0] += learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden[0])

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        for i in range (0, self.nL):
          print("The final weights of hidden layer {i} is {self.W_hidden[i]}.")
          print("The final biases of hidden layer {i} is {self.Wb_hidden[i]}.")
        print("The final weight of output layer is {self.W_output}.")
        print("The final biases of output layer {self.Wb_output}.")

    def forward_pass(self, xValue=0, activation="sigmoid"):
        # pass our inputs through our neural network
        self.X_hidden = []
        
        for ind in range(self.nL):
            if ind == 0:
                in_hidden = np.dot(xValue, self.W_hidden[ind]) + self.Wb_hidden[ind]
            else:
                in_hidden = np.dot(self.X_hidden[ind-1], self.W_hidden[ind]) + self.Wb_hidden[ind]
                
            if activation == "sigmoid":
                self.X_hidden.append(self.__sigmoid(in_hidden))
            elif activation == "tanh":
                self.X_hidden.append(self.__tanh(in_hidden))
            elif activation == "relu":
                self.X_hidden.append(self.__relu(in_hidden))

        in_output = np.dot(self.X_hidden[self.nL-1], self.W_output) + self.Wb_output  # TO DO: in hidden two, then in output afterwards

        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        elif activation == "tanh":
            out = self.__tanh(in_output)
        elif activation == "relu":
            out = self.__relu(in_output)
        return out

    def backward_pass(self, out, activation): #this is more the delta calculation than backward pass
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        elif activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        elif activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation="sigmoid"):
        # the last hidden layer uses the weights of output
        if activation == "sigmoid":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden[self.nL - 1]))
        elif activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__tanh_derivative(self.X_hidden[self.nL - 1]))
        elif activation == "relu":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__relu_derivative(self.X_hidden[self.nL - 1]))
        self.deltaHidden[self.nL - 1] = delta_hidden_layer

        # the layers (other than last) are calculated using other deltas
        for i in range(self.nL - 2, -1, -1):
          if activation == "sigmoid":
              delta_hidden_layer = (self.deltaHidden[i+1].dot(self.W_hidden[i+1].T)) * (self.__sigmoid_derivative(self.X_hidden[i]))
          elif activation == "tanh":
              delta_hidden_layer = (self.deltaHidden[i+1].dot(self.W_hidden[i+1].T)) * (self.__tanh_derivative(self.X_hidden[i]))
          elif activation == "relu":
              delta_hidden_layer = (self.deltaHidden[i+1].dot(self.W_hidden[i+1].T)) * (self.__relu_derivative(self.X_hidden[i]))
          self.deltaHidden[i] = delta_hidden_layer



    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, activation="sigmoid", header=True):
        # TODO: obtain prediction on self.test_dataset
        self.yPredict = self.forward_pass(self.xTest, activation)
        #self.yPredict = self.yPredict.flatten()
        diff = self.yPredict - self.yTest
        testError = 0.5 * np.sum(np.square(diff), axis=0)
        return testError

    def preprocessData(self, datafile):
        df = pd.read_csv(datafile, header='infer', delimiter=",", na_values=[" "])
        # Drop empty rows i.e. rows with " "
        df = df.dropna()

        # Columns desciption:
        # Pixels (0-447) | Motion type (448)
        lastCol = 448  # Column containing classification values
        self.totClassifiers = 4  # Total Binary classifiers

        # Convert motion type to binary values:
        motionCopy = df[str(lastCol)]
        motionCopy2 = motionCopy.copy()
        for i in range(self.totClassifiers):
            value = np.zeros(self.totClassifiers)
            value[i] = 1
            df[str(lastCol + i)] = motionCopy
            df[str(lastCol + i)] = df[str(lastCol + i)].replace(to_replace="L", value=value[0], inplace=False)
            df[str(lastCol + i)] = df[str(lastCol + i)].replace(to_replace="F", value=value[1], inplace=False)
            df[str(lastCol + i)] = df[str(lastCol + i)].replace(to_replace="R", value=value[2], inplace=False)
            df[str(lastCol + i)] = df[str(lastCol + i)].replace(to_replace="S", value=value[3], inplace=False)
            
            # Data is not clean since lowercase s existed
            df[str(lastCol + i)] = df[str(lastCol + i)].replace(to_replace="s", value=value[3], inplace=False)
            motionCopy = motionCopy2.copy()

        x = df.iloc[:, 0:lastCol]  # Extract x values from data frame
        y = df.iloc[:, lastCol:lastCol + self.totClassifiers + 1]  # Extract y values from data frame

        #from sklearn.model_selection import train_test_split
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=0.80)  # Add random_state = 3 to get consistent data similar to the report

        # Compute sde, mean of the data
        scaler = StandardScaler()
        scaler.fit(xTrain)

        # Transform the x data
        xTrain = scaler.transform(xTrain)
        xTest = scaler.transform(xTest)

        # Convert y data to lists
        yTrain = yTrain.to_numpy()
        yTest = yTest.to_numpy()

        return xTrain, xTest, yTrain, yTest

    # Converts the predicted 2D array for multi binary classification to string representation
    def postProcess(self):
        outVal = np.array(["L", "F", "R", "S"])
        self.yTestString = ["" for ind in range(len(self.yTest))]
        self.yPredictString = ["" for ind in range(len(self.yPredict))]
        diff = 0

        for ind in range(len(self.yTest)):
            self.yTestString[ind] = outVal[np.argmax(self.yTest, axis=1)[ind]]
            self.yPredictString[ind] = outVal[np.argmax(self.yPredict, axis=1)[ind]]

            if not self.yPredictString[ind] == self.yTestString[ind]:
                diff = diff + 1

        return diff

    def preprocessDataAssign(self,datafile):
        df = pd.read_csv(datafile,header=None,delimiter = ",",na_values=[" "])
        # Drop empty rows i.e. rows with " "
        df = df.dropna()
        
        # Columns desciption:
        # Front | Left | Right | Back | Motion type
        lastCol        = 4 # Column containing classification values
        self.totClassifiers = 4 # Total Binary classifiers 
        
        # Convert motion type to binary values:
        tempCol = df[[lastCol]]
        for i in range(self.totClassifiers):
            value           = np.zeros(self.totClassifiers)
            value[i]        = 1
            df[[lastCol+i]] = tempCol
            df[[lastCol+i]] = df[[lastCol+i]].replace(to_replace = "Move-Forward",      value = value[0])
            df[[lastCol+i]] = df[[lastCol+i]].replace(to_replace = "Slight-Right-Turn", value = value[1])
            df[[lastCol+i]] = df[[lastCol+i]].replace(to_replace = "Slight-Left-Turn",  value = value[2])
            df[[lastCol+i]] = df[[lastCol+i]].replace(to_replace = "Sharp-Right-Turn",  value = value[3])

        
        x = df.iloc[:,0:lastCol] # Extract x values from data frame
        y = df.iloc[:,lastCol:lastCol + self.totClassifiers+1]   # Extract y values from data frame
        
        #from sklearn.model_selection import train_test_split
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size = 0.80, random_state=3) # Add random_state = 3 to get consistent data similar to the report
            
        # Compute sde, mean of the data  
        scaler = StandardScaler()
        scaler.fit(xTrain)
            
        # Transform the x data
        xTrain = scaler.transform(xTrain)
        xTest  = scaler.transform(xTest)
                  
        # Convert y data to lists
        yTrain = yTrain.to_numpy()
        yTest  = yTest.to_numpy()
        
        return xTrain, xTest, yTrain, yTest
    
    # Converts the predicted 2D array for multi binary classification to string representation
    def postProcessAssign(self):
        outVal = np.array(["Move-Forward", "Slight-Right-Turn", "Slight-Left-Turn", "Sharp-Right-Turn"])
        self.yTestString = ["" for ind in range(len(self.yTest))]
        self.yPredictString = ["" for ind in range(len(self.yPredict))]
        diff = 0
        
        for ind in range(len(self.yTest)):
            self.yTestString[ind] = outVal[np.argmax(self.yTest, axis = 1)[ind]]
            self.yPredictString[ind] = outVal[np.argmax(self.yPredict, axis = 1)[ind]]
            
            if not self.yPredictString[ind] == self.yTestString[ind]: 
               diff = diff + 1
        
        return diff

if __name__ == "__main__":
    # perform pre-processing of both training and test part of the test_dataset
    # split into train and test parts if needed
    #preprocessData("https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_24.data")
    #neural_network = NeuralNet("https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Project/autonomous_arena.csv")
    neural_network = NeuralNet("https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_4.data")
    activationFunc = "sigmoid"  # "sigmoid" "tanh" or "relu"
    neural_network.train(activationFunc, max_iterations=5000, learning_rate=0.001)
    testError = neural_network.predict(activation=activationFunc)
    print("Test error = " + str(testError))

    # Convert predicted values back to string
    testError = neural_network.postProcessAssign()
    print("First 30 values:\n")
    print("Predicted Motion \t Actual Motion\n")
    for ind in range(30):
        print("%-17s \t %s" % (neural_network.yPredictString[ind], neural_network.yTestString[ind]))
    print("Number of predicted classifications that differed from actual values: " + str(testError))
