import scipy.io
import numpy as np
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, k):
        """
        :param k: The number of clusters needed
        """
        self.k = k
        self.centroids = None
        self.clusters = {}
        self.obj_fun_val = 0

    def initialize_centers(self, train_x, init_strategy):
        """
        :param train_x: Input samples numpy array
        :param init_strategy: Strategy to initialize the centroids.
        :return:
        """
        if init_strategy == 2:
            self.init_strategy_2(train_x)
        elif init_strategy == 1:
            self.init_strategy_1(train_x)

    def init_strategy_1(self, train_x):
        """
        Initialize according to first strategy.
        i.e., choose K random centroids from the input samples without replacement.
        :param train_x: Input samples numpy array
        :return:
        """
        self.centroids = np.array(train_x[np.random.choice(train_x.shape[0], self.k, replace=False)])

    def init_strategy_2(self, train_x):
        """
        Initialize according to second strategy.
        Choose first centroid randomly from the input samples.
        For centroid i, we calculate distance of points from all i-1 centroids before.
        Point with largest average distance becomes centroid i.
        :param train_x: Input samples numpy array
        :return:
        """
        # choosing first centroid randomly from input samples.
        centroids = [train_x[np.random.choice(train_x.shape[0])]]
        for index in range(1, self.k):
            distances = np.zeros(train_x.shape[0])
            # for centroid i, loop through centroids i-1, ..., 1.
            for j in range(index - 1, -1, -1):
                # calculate distance of input sample with centroid j
                distances = np.add(distances, np.sum((train_x - centroids[j]) ** 2, axis=1))
            # point with largest distance from all the centroids (1,..., i-1) chosen as new centroid i.
            index = np.argmax(distances)
            centroids.append(train_x[index])
        # store centroids as class attribute to use it in process ahead.
        self.centroids = np.array(centroids)

    def cluster(self, train_x):
        """
        Classify examples into the cluster with whom its distance is minimum.
        After classifying, Update the newly formed clusters' centroid.
        :param train_x: Input samples numpy array
        :return:
        """
        same_ctr = False
        while not same_ctr:
            # store the old centroids array for comparison.
            old_ctr = self.centroids.copy()
            # creating empty clusters for next iteration
            for index in range(self.k):
                self.clusters[index] = []
            # Go through each training example and calculate it's distance from centroids.
            # Assign point to the clusters whose centroid and point has minimum distance.
            for example in train_x:
                min_dist_ctr = np.argmin(np.sum((self.centroids - example) ** 2, axis=1))
                self.clusters[min_dist_ctr].append(example)
            # Now go through each cluster and update the centroid of the cluster with newly formed
            # cluster's centroid(mean).
            for key in self.clusters.keys():
                # numpy mean throws a error if array as no elements in it. So if condition to check that array is not
                # an empty array
                if self.clusters[key]:
                    self.centroids[key] = np.mean(np.array(self.clusters[key]), axis=0)
            # convergence condition: see if old centroid and new centroids are equal
            # If they are equal then stop the algorithm else continue updating centroids.
            if np.array_equal(old_ctr, self.centroids):
                same_ctr = True
                # if convergence is there, count the objective function and store it as a attribute.
                self.count_obj_fun()

    def count_obj_fun(self):
        """
        calculate objective function after convergence.
        :return:
        """
        for key in self.clusters.keys():
            if self.clusters[key]:
                self.obj_fun_val += np.sum(np.array((self.clusters[key]) - self.centroids[key]) ** 2)


def load_data(file_path):
    """
    Read the input file with .mat extension.
    :param file_path:
    :return:
    """
    data = scipy.io.loadmat(file_path)
    return data['AllSamples']


def prepare_input(file_path):
    """
    Loading data, finding 2 features per raw and normalizing data
    :param file_path: input file path
    :return: training input samples in numpy array
    """
    # reading data from file
    train_x = load_data(file_path)
    return train_x


def run_kmeans_clustering(train_x, run, strategy, colors, labels):
    xAxis, yAxis = [], []
    for k in range(2, 11):
        # for each cluster
        # run the k-means clustering
        kmeans = KMeans(k)
        # initialize the centroids according to strategy
        kmeans.initialize_centers(train_x, init_strategy=strategy)
        # classify the points and update the centroids
        kmeans.cluster(train_x)
        print("For K =", k, " Objective value function=", kmeans.obj_fun_val)
        # save data to plot the graphs
        xAxis.append(k)
        yAxis.append(kmeans.obj_fun_val)
        # printing number of nodes in each centroid at convergence for k such centroids.
        for i in range(k):
            print(kmeans.centroids[i], len(kmeans.clusters[i]), end="\n")
        print("\n")
        # Plotting the points cluster-wise and centroid of respective clusters
        for c in range(kmeans.centroids.shape[0]):
            plt.scatter(np.array(kmeans.clusters[c])[:, 0], np.array(kmeans.clusters[c])[:, 1],
                        c=colors[c], label=labels[c])
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=200, c='yellow', label='ctr')
        plt.title('Clusters')
        plt.legend()
        plt.savefig('RUN_' + str(run) + '__Strategy_' + str(strategy) + str(k) + '.jpg')
        plt.clf()
    # plotting objective function vs number of clusters for elbow method analysis
    plt.plot(xAxis, yAxis)
    plt.xlabel('Number of Clusters')
    plt.ylabel('within-cluster sums of squares (WCSS)')
    plt.title('Elbow method to determine optimum number of clusters')
    plt.savefig('RUN_' + str(run) + '__Strategy_' + str(strategy) + '.jpg')
    plt.clf()


def cluster_and_plot(train_x, *args, **kwargs):
    for run in range(1, 3):
        # for each run
        for strategy in range(1, 3):
            # for each strategy
            run_kmeans_clustering(train_x, run, strategy, *args, **kwargs)


if __name__ == "__main__":
    path = "AllSamples.mat"
    trainX = prepare_input(path)
    color = ['red', 'blue', 'green', 'cyan', 'magenta', 'pink', 'lime', 'darkorange', 'gray', 'black', 'mintcream']
    label = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
    cluster_and_plot(trainX, colors=color, labels=label)
