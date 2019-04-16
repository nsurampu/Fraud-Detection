import pandas as pd
import numpy as np
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import operator
from collections import OrderedDict
import itertools
import time
from sklearn.metrics import accuracy_score
import visualization

#   Creating Visualization() object for calling methods
visuals = visualization.Visualization()

class Local_Outlier_Factor:
    """
    This class contains methods that are used for calculating the required parameters
    and values for the LOF of points in a given dataset. The points are accordingly marked
    as outliers depending on the threshold value set.

    Attributes
    ----------
    K : int
        The number of points in the neighborhood of a give point.
    DATA : pandas dataframe
        Dataset to be operated upon.
    SAMPLE_DATA : list
        A sample set of 2-D points.
    DATA_FLAG : bool
        A flag for triggering usage of actual/sample dataset.
    THRESH : float
        Threshold value for marking outliers.
    REDUCED_POINTS : list
        List of dimensionally reduced points.

    """

    def __init__(self):
        self.K = 2
        self.DATA = None
        self.SAMPLE_DATA = None
        self.DATA_FLAG = True
        self.THRESH = 1
        self.REDUCED_POINTS = []

    def neighborhood(self):
        """
        A method that performs operations required for calculating the LOFs of points in
        a some neighborhood.

        Returns
        -------
        type list
            A list consisting of LRDs and neighbors of points.

        """
        if self.DATA_FLAG:
            val_data = self.DATA.values.tolist()
        else:
            val_data = self.SAMPLE_DATA   # for sample sets
        lrds = []
        reach_distances = []
        read_index1 = 0
        neighbors_dict = {}
        reduced_points = []
        for reading1 in val_data:
            self.REDUCED_POINTS.append(visuals.dimension_reduction(reading1))
            neighbors = {}
            neighbors_dict[read_index1] = []
            read_index2 = 0
            for reading2 in val_data:
                if read_index1 != read_index2:
                    print("Reading indices: " + str(read_index1) + " " + str(read_index2))
                    distance = sum(abs(np.array(list(reading1)) - np.array(list(reading2))))
                    distance = round(distance, ndigits=2)
                    neighbors[read_index2] = distance
                read_index2 = read_index2 + 1
            sorted_temp = sorted(neighbors.items(), key=lambda kv: kv[1])
            neighbors = OrderedDict(sorted_temp)
            neighbors = list(itertools.islice(neighbors.items(), 0, self.K))
            # print(neighbors)
            for n in neighbors:
                neighbors_dict[read_index1].append(n)
            lrds.append(self.LRD(neighbors, self.K))
            read_index1 = read_index1 + 1

        return [lrds, neighbors_dict]

    def K_element_dist(self, read_index1, K):
        """
        A method that calculates the distance of the Kth neighbor.

        Parameters
        ----------
        read_index1 : int
            Index of point to be used..
        K : int
            Number of neighbors.

        Returns
        -------
        type float
            Distance of Kth neighbor for passed point.

        """
        if self.DATA_FLAG:
            val_data = self.DATA.values.tolist()
        else:
            val_data = self.SAMPLE_DATA
        k_dists = []
        reading1 = val_data[read_index1]
        read_index2 = 0
        for reading2 in val_data:
            if read_index1 != read_index2:
                distance = sum(abs(np.array(list(reading1)) - np.array(list(reading2))))
                distance = round(distance, ndigits=2)
                k_dists.append(distance)
            read_index2 = read_index2 + 1

        k_dists.sort()
        k_dists = k_dists[0:self.K]
        # print(k_dists)
        return k_dists[-1]

    def LRD(self, neighbors, K):
        """
        THis method calculates the LRD of a given point.

        Parameters
        ----------
        neighbors : list
            List of neighbors and their distances from the given point.
        K : type
            Number of neighbors.

        Returns
        -------
        type float
            LRD of the given point.

        """
        k_nearest_count = len(neighbors)
        reach_distance_sum = self.reach_distance(neighbors, self.K)
        lrd = k_nearest_count / reach_distance_sum

        return lrd

    def reach_distance(self, neighbors, K):
        """
        This method calculates the reach distance of a point from it's neighbors.

        Parameters
        ----------
        neighbors : list
            List of neighbors and their distances from the given point.
        K : type
            Number of neighbors.

        Returns
        -------
        type float
            The total reach distance of a point from it's neighbors.

        """
        rds = []
        for element in neighbors:
            rd = max(self.K_element_dist(element[0], self.K), element[1])
            rds.append(rd)

        return sum(rds)

    def LOF(self, lrds, neighbors_dict, K):
        """
        This method calculates the LOF of a point and determines if the point is an
        outlier or not.

        Parameters
        ----------
        lrds : list
            List of LRDs of points.
        neighbors_dict : dict
            Dictionary of points and their respective K neighbors.
        K : int
            Number of neighbors.

        Returns
        -------
        type list
            List of LOFs of points.

        """
        lofs = []
        # print(neighbors_dict)
        for element in neighbors_dict.keys():
            print("Calculating LOF for: " + str(element))
            neighbors = neighbors_dict[element]
            lrd_sum = 0
            reach_dist_sum = self.reach_distance(neighbors, self.K)
            for n in neighbors:
                lrd_sum = lrd_sum + lrds[n[0]]
                # reach_dist_sum = reach_dist_sum + reach_distances[n]
            lof = (lrd_sum * reach_dist_sum) / (self.K**2)
            lof = round(lof, ndigits=2)
            # specific for fraud detection
            if lof > self.THRESH:
                lof = 1
                visuals.OUTLIERS.append(self.REDUCED_POINTS[element])
            else:
                lof = 0
                visuals.NON_OUTLIERS.append(self.REDUCED_POINTS[element])
            lofs.append(lof)

        return lofs

    def container(self):
        """
        This method acts a container method for calling all the required methods together
        while operating in a thread pool.

        Returns
        -------
        type list
            List of LOFs of points.

        """
        lof_reqs = self.neighborhood()
        lofs = self.LOF(lof_reqs[0], lof_reqs[1], self.K)

        return lofs

if __name__ == "__main__":

    lof_class = Local_Outlier_Factor()

    credit_data = pd.read_csv('../creditcard_normalized.csv')

    y = credit_data['Class']

    req_cols = []
    for i in range(1, 29):
        req_cols.append('V' + str(i))

    req_cols.append('Time')
    req_cols.append('Amount')

    data = credit_data[req_cols]
    sample_data = [[0,0],[0,1],[1,1],[3,0]]   # some sample data

    n = 100

    lof_class.DATA = data[0:n]
    val_y = y[0:n]

    lof_class.SAMPLE_DATA = sample_data
    lof_class.DATA_FLAG = True
    if lof_class.DATA_FLAG:
        lof_class.K = 5
        lof_class.THRESH = 1.5

    visuals.K = lof_class.K

    pool = ThreadPool(processes=cpu_count())

    start_time = time.clock()

    lofs = (pool.apply_async(lof_class.container)).get()

    stop_time = time.clock()
    run_time = stop_time - start_time

    if lof_class.DATA_FLAG:
        print("Accuracy: " + str(accuracy_score(lofs, val_y)))

    print("Time: " + str(run_time))
    visuals.outlier_plot()
