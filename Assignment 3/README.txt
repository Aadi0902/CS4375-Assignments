README
Authors: Aadi Kothari and Andrew Su


Directions:
1. Open Kmeans.py
2. You can adjust K or the dataset used in line 140 in the main method.
4. Run Kmeans.py

Files:
1. Kmeans.py - This file contains the entire project, including the preprocessing and finding kmeans
2. online files - hosted on github
  a.) kmeans = Kmeans("https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Assignment%203/Datasets/small.txt", 2)
    This is a test dataset is a dummy dataset with simple tweets and obvious centroids
  b.) kmeans = Kmeans("https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Assignment%203/Datasets/usnewshealth.txt", 256)
    These are tweets from U.S. News Helath.



Libraries:
numpy (numpy.org)
pandas (pandas.pydata.org)
sklearn (scikit-learn.org)
random - used for randomly generating new centroids


Dataset:
US News Health tweets

Dataset link:
https://raw.githubusercontent.com/Aadi0902/CS4375-Machine-Learning-Assignments/master/Assignment%203/Datasets/usnewshealth.txt

This dataset contains tweet data from US News Health section.  There is additional data like tweet number and urls that are removed
from in preprocessing.
