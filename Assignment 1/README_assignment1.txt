README


Directions:
1. Store the given files “Part1.py”, “Part2.py” and “preProcessing.py” in one folder
2. Run Part1.py
3. Run Part2.py



Files:
Part1.py - Contains the solution for Part 1, using our own gradient descent algorithm
Part2.py - Contains the solution for Part 2, using a gradient descent algorithm from Scikit Learn package.
preProcessing.py - Contains code that normalizes and splits the data. This file does not need to be run directly.  Part1.py and Part2.py both import and run the functions needed.


Libraries:
numpy (numpy.org)
pandas (pandas.pydata.org)
sklearn (scikit-learn.org)
matplotlib (matplotlib.org)
preProcessing (preprocessing.py)


Dataset:
Auto MPG Data Set from UCI Machine Learning Repository

Dataset link:
archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data

This data set contains 398 instances of fuel economy (in MPG) vehicles from 1970-1980,
along with related features like engine displacement, horsepower, and weight.
Attributes:
1. mpg
2. cylinders
3. displacement
4. horsepower
5. weight
6. acceleration
7. model year
8. origin
9. car name

For our processing, we do not use origin because it was categorical data, and car name was also not used because we concluded that it wasn’t relevant to calculating mpg.

Additional Dataset information:
archive.ics.uci.edu/ml/datasets/Auto+MPG
