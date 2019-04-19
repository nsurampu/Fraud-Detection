# Fraud Detection using Local Outlier Factor

This implementation of **Local Outlier Factor (LOF)** attempts to detect frauds in a given database of credit card transactions.

The database can be found here: ![Credit Card Fraud Detection (Kaggle)](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The project is divided into three major parts
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

The default threshold value is set to **1.5** for the credit data, while it is **1** for a sample data of four 2-D points.
The threshold value can be changed using ***THRESH*** and the sample data can be toggled using ***DATA_FLAG***.
The default value for K is set to **2** which can be changes using ***K***.

**Visualization** involves applying dimensional reduction to the points and reduce the number of attributes to 2. This helps in plotting on a 2-D graph with points divided into two classes. The reduction is done using a modulo-i operation, where ***i***
is the index of the point in the dataset.

## Results

The current implementation of LOF using a threshold of 1.5 and K = 5 consistently gives an accuracy of above **85%** over various permutations of the dataset, with an average accuracy of **93%**.

An sample run of the dataset on the first **500** samples gives:
<br>**Accuracy: 96%**
<br>**Run Time: 15 seconds**

**Scatter plot after dimensional reduction:**

![Scatter Plot (1000 samples)](https://github.com/nsurampu/Fraud-Detection/blob/master/sample_run(500).png)


## Author
![Naren Surampudi](https://github.com/nsurampu/)
