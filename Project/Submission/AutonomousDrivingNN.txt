#####################################################################################################################
#   Project: Deep Neural Network Programming
#   Please install numpy and pandas before running the following code
#   Symbol meanings have been described below
#   The init method of NeuralNet class takes file path as parameter and splits it into train and test part
#         - it assumes that the last column has the label (output) column
#   h             - array for number of neurons in each hidden layer
#   X             - vector of features for trainging instances
#   xTest         - vector of features for testing instances
#   y             - output for each training instance
#   yTest         - output for each testing instance in integer array form
#   yTestString   - ouptput for each testing instance in String form (Type of motion)
#   yPredict      - output for each predicted instacne in integer array form
#   yPredctString - output for each predicted instance in String form (Type of motion)
#   W_hidden      - list of weight matrix connecting input to hidden layers
#   Wb_hidden     - list of bias matrix for the hidden layers
#   W_output      - weight matrix connecting hidden layer to output layer
#   Wb_output     - bias matrix connecting hidden layer to output layer
#   deltaOut      - delta for output unit (see slides for definition)
#   deltaHidden   - delta for hidden unit (see slides for definition)
#   X_hidden      - X values in the hidden layers
#   in_hidden     - values right before passing through activation function
#   other symbols have self-explanatory meaning
#
#####################################################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NeuralNet:
    def __init__(self, dataFile, header=True, h=[80, 30, 10]):  # values of h array correspond to nodes in respective hidden layer
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset

        self.nHL = len(h) # Number of hidden layers
        self.X, self.xTest, self.y, self.yTest = self.preprocessData(dataFile) # Preprocessing and train-test split

        # Find number of input and output layers from the dataset
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
       
        # Create a 3D array here for inner layers
        # Create list of W_hidden and Wb_hidden, 1 for each hidden layer
        self.W_hidden = [] # List of matricies for hidden layer weights
        self.Wb_hidden = [] # List of matricies for hidden layer biases

        # Initialize first hidden layer weights to random values between 1 and -1
        self.W_hidden.append(2 * np.random.random((input_layer_size, h[0]))-1)
        self.Wb_hidden.append(2 * np.random.random((1, h[0]))-1)

        # Initialize other hidden layer weights to random values between 1 and -1
        for i in range(1, self.nHL):
            self.W_hidden.append(2 * np.random.random((h[i - 1], h[i]))-1)
            self.Wb_hidden.append(2 * np.random.random((1, h[i]))-1)

        # Initialize output layer weights and biases
        self.W_output = 2 * np.random.random((h[self.nHL - 1], self.output_layer_size))-1
        #self.Wb_output = np.ones((1, self.output_layer_size))
        self.Wb_output = 2 * np.random.random((1, self.output_layer_size)) -1

        # Initialize output layer delta values to zeroes
        self.deltaOut = np.zeros((self.output_layer_size, 1))
        ##self.deltaHidden = np.zeros((h, 1))

        # Initialize hidden layer delta values to zeroes
        self.deltaHidden = []
        for num_nodes in h:
            self.deltaHidden.append(np.zeros((num_nodes, 1)))
        # Assign h to a global variable hidden layers
        self.hidden_layers = h

    # Define activation functions
    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)

    # Define derivative of activation functions
    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)


    # Define individual activation functions
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))


    # Define individual deivatives
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return (1 - x**2)

    # Training function
    def train(self, activation="sigmoid", max_iterations=5000, learning_rate=0.001):

        for iteration in range(max_iterations): # Go through all the iterations

            out = self.forward_pass(self.X, activation) # Perdorm forward pass
            error = 0.5 * np.power(out - self.y, 2) # Calculate mean square error
            self.backward_pass(out, activation, learning_rate)
            
        print("After " + str(max_iterations)+" iterations, the total error is " + str(np.sum(error)))
        # for i in range(0, self.nHL):
        #     print(f"The final weights of hidden layer {i} is {self.W_hidden[i]}.")
        #     print(f"The final biases of hidden layer {i} is {self.Wb_hidden[i]}.")
        # print(f"The final weight of output layer is {self.W_output}.")
        # print(f"The final biases of output layer {self.Wb_output}.")

    # Perform forward pass
    def forward_pass(self, xValue=0, activation="sigmoid"):

        self.X_hidden = []

        for ind in range(self.nHL): # Iterate through hidden layers
            if ind == 0: # 1st iteration is different as input x values are used
                in_hidden = np.dot(xValue, self.W_hidden[ind]) + self.Wb_hidden[ind]
            else:
                in_hidden = np.dot(self.X_hidden[ind - 1], self.W_hidden[ind]) + self.Wb_hidden[ind]

            if activation == "sigmoid":
                self.X_hidden.append(self.__sigmoid(in_hidden))
            elif activation == "tanh":
                self.X_hidden.append(self.__tanh(in_hidden))

        # Output before passing through activation function
        in_output = np.dot(self.X_hidden[self.nHL - 1], self.W_output) + self.Wb_output

        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        elif activation == "tanh":
            out = self.__tanh(in_output)

        return out
    
    # Perform backward pass
    def backward_pass(self, out, activation, learning_rate):
      self.delta_calculation(out, activation)
      self.weights_calculation(learning_rate)

    # Calculate delta values
    def delta_calculation(self, out, activation):
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)
    
    # Calulate weights
    def weights_calculation(self, learning_rate = 0.001):
        # Calculate W_output last X_hidden - x value of last hidden layer
        self.W_output += learning_rate * np.dot(self.X_hidden[self.nHL - 1].T, self.deltaOut)
        self.Wb_output += learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)

        for i in range(1, self.nHL): # Iterate through hidden layers to assign weights
          self.W_hidden[i] += learning_rate * np.dot(self.X_hidden[i - 1].T, self.deltaHidden[i])
          self.Wb_hidden[i] += learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden[i])

        # Assign first hidden layer weights - done separately as training values are used here
        self.W_hidden[0] += learning_rate * np.dot(self.X.T, self.deltaHidden[0])
        self.Wb_hidden[0] += learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden[0])

    # Compute output delta values
    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = np.multiply((self.y - out), self.__sigmoid_derivative(out)) # Element wise multiplication
        elif activation == "tanh":
            delta_output = np.multiply((self.y - out), self.__tanh_derivative(out)) # Element wise multiplication

        self.deltaOut = delta_output

    # Compute hidden delta values
    def compute_hidden_delta(self, activation="sigmoid"):
        # the last hidden layer uses the weights of output
        if activation == "sigmoid":
            delta_hidden_layer = np.multiply((self.deltaOut.dot(self.W_output.T)), (self.__sigmoid_derivative(self.X_hidden[self.nHL - 1])))
        elif activation == "tanh":
            delta_hidden_layer = np.multiply((self.deltaOut.dot(self.W_output.T)), (self.__tanh_derivative(self.X_hidden[self.nHL - 1])))
        self.deltaHidden[self.nHL - 1] = delta_hidden_layer

        # the layers (other than last) are calculated using other deltas
        for i in range(self.nHL - 2, -1, -1):
            if activation == "sigmoid":
                delta_hidden_layer = np.multiply((self.deltaHidden[i + 1].dot(self.W_hidden[i + 1].T)), (self.__sigmoid_derivative(self.X_hidden[i])))
            elif activation == "tanh":
                delta_hidden_layer = np.multiply((self.deltaHidden[i + 1].dot(self.W_hidden[i + 1].T)), (self.__tanh_derivative(self.X_hidden[i])))

            self.deltaHidden[i] = delta_hidden_layer            

    # Predict function - returns test error
    def predict(self, activation="sigmoid", header=True):
        self.yPredict = self.forward_pass(self.xTest, activation)
        diff = self.yPredict - self.yTest
        testError = 0.5 * np.sum(np.square(diff), axis=0)
        return testError

    # Preprocess the given data
    def preprocessData(self, datafile):
        df = pd.read_csv(datafile, header='infer', delimiter=",", na_values=[" "])
        # Drop rows with empty spaces i.e. rows with " "
        df = df.dropna()

        # Columns desciption:
        # Pixels (0-447) | Motion type (448)
        lastCol = 448  # Column containing classification values
        self.totClassifiers = 4  # Total Binary classifiers defined as L | F | R | S
        # L, F, R, S corresponds to Left, Front, Right, Stop

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
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=0.80, random_state = 3)  # Add random_state = 3 to get consistent data similar to the report

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
        # Output value array
        outVal = np.array(["L", "F", "R", "S"])
        self.yTestString = ["" for ind in range(len(self.yTest))]
        self.yPredictString = ["" for ind in range(len(self.yPredict))]
        diff = 0 # Number of test data that did not match predicted data

        for ind in range(len(self.yTest)):
            self.yTestString[ind] = outVal[np.argmax(self.yTest, axis=1)[ind]]
            self.yPredictString[ind] = outVal[np.argmax(self.yPredict, axis=1)[ind]]

            if not self.yPredictString[ind] == self.yTestString[ind]:
                diff = diff + 1

        return diff


if __name__ == "__main__":
    # perform pre-processing of both training and test part of the test_dataset
    # split into train and test parts if needed
    neural_network = NeuralNet("https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Project/autonomous_arena.csv", h=[80, 30, 10])
    #unmodified: https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Project/autonomous_arena.csv
    #modified: https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Project/autonomous_arenaModified.csv
    activationFunc = "sigmoid"  # "sigmoid" "tanh"
    neural_network.train(activationFunc, max_iterations=3000, learning_rate=0.001)
    testError = neural_network.predict(activation=activationFunc)
    print("Test error = " + str(testError))

    # Convert predicted values back to string
    testError = neural_network.postProcess()
    print("First 30 values:\n")
    print("Predicted Motion \t Actual Motion\n")
    for ind in range(0, 300, 6):
        print("%-17s \t %s" % (neural_network.yPredictString[ind], neural_network.yTestString[ind]))

    # for ind in range(0, 300, 6):
    #     print("%-17s \t %s" % (neural_network.yPredict[ind],neural_network.yTest[ind]))
    print("Number of predicted classifications that differed from actual values: " + str(testError))
