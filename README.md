# Fraud Detection using Local Outlier Factor

This implementation of **Local Outlier Factor (LOF)** attempts to detect frauds in a given database of credit card transactions.

The database can be found here: ![Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## The project is divided into three major parts
1. Preprocessing
2. LOF Calculation and Fraud Detection
3. Visualization

**Preprocessing** mainly involves reading the data provided in the form of a CSV file and normalizing the data to
make it more ***smooth***. This helps us improve LOF calculation results.

**LOF Calculation** is the heart of the project and involves calculating the LOF score of each point, comparing it with a
custom threshold value and concluding if a point represents an outlier or not- in this case a fraud. This involves the following main steps:
1. Finding K nearest neighbors
2. Finding the Kth neighbor for a point and it's distance.
3. Calculating the ***Reach Distance (RD)*** for a point with respect to it's K neighbors.
4. Calculating ***Local Reachability Density (LRD)*** of the point.
5. Calculating ***Local Outlier Factor (LOF)*** of point using RD and LRD.
6. Comparing LOF with a threshold value. If LOF > threshold, then the point most probably is an outlier/fraud.

The default threshold value is set to **1.5** and can be changed in the ***LOF.py*** script by changing the ***THRESH*** class
variable.

**Visualization (Work in Progress)** involves applying dimensional reduction to the points and reduce the number of attributes to 2. This helps in plotting on a 2-D graph with points divided into two classes.

## Results

The current implementation of LOF using a threshold of 1.5 consistently gives an accuracy of above **85%** over various
permutations of the dataset, with an average accuracy of **93%**.

## Author
![Naren Surampudi](https://github.com/nsurampu/)
