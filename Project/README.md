README
Authors:
Rami Mohammed Ismael
Laila Ashraf Kamel
Aadi M Kothari
Andrew Su

Date Updated: 11/24/2020

Directions:
1. Open DoublePossibleNasm.py, this is the only file that contains code.
2. The following hyperparameters can be tuned:
    a. Shape of hidden layers - in line 298 {h} can be set to an array of integers that represent the nodes in each hidden layer. Default is [80, 30, 10]
    b. Number of iterations - in line 302 {max_iterations} can be set. Default is 5000.
    c. Learning rate - in line 302 {learning_rate} can be set. Default is .0001.
3. Run DoublePossibleNasm.py. This will take a few minutes on the main dataset, autonomous_arena, depending on the number of iterations and the complexity of the hidden layers.

Files:
1. DoublePossibleNasm.py - This file contains the entire project, including preprocessing, training, and applying on the neural net on the test set.
2. online files - hosted on github
  a.) original dataset: https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Project/autonomous_arena.csv


Libraries:
numpy (numpy.org)
pandas (pandas.pydata.org)
sklearn (scikit-learn.org)
random - used for randomly generating new centroids

Dataset:
Data collected from an autonomous car project.  Grayscale pixel values are matched with corresponding movements: Left, Right, Forward, Stop

Kaggle page link:
https://www.kaggle.com/firstofhisname/indoor-car-track
Dataset Link:
https://www.kaggle.com/firstofhisname/indoor-car-track/download
