import time

import scipy.io
import numpy as np


class LogisticReg:
    def __init__(self):
        """
        Setting up attributes for Logistic Regression.
        w: weight vector for LR
        b: bias scalar for LR
        eta: Learning rate
        eps: epsilon to add to prediction(y_pred) to avoid log(0) error.
        """
        self.eta = 5
        self.w = None
        self.b = 0
        self.eps = 1e-6

    def weight_init(self, n_features):
        """
        :param n_features: Number of features in input array. Here it's 2.
        :return:
        """
        self.w = np.zeros(n_features)
        self.b = 0

    def train(self, X, Y, iterations):
        """
        The training loop for LR.
        1.We calculate Y_pred = W*X
        2.Find likelihood of Y_pred
        3.derivation of lieklihood with respect to W
        4.Update the W vector and b(bias) with the gradient
        5.log the results, loss and return it.

        :param X: Input vector of shape (n_examples, n_features)
        :param Y: Ouput vector of shape (n_examples)
        :param iterations: Iterations for gradient ascent
        :return: Likeihood at each iteration. Just like a loss, It should go near 0.
        """
        loss_summary = []
        self.weight_init(X.shape[1])
        for i in range(iterations):
            # Y = W*X
            y_pred = sigmoid(np.dot(self.w, X.T) + self.b)
            # Adding eps to not encounter log(0)
            y_pred = np.maximum(np.full(y_pred.shape, self.eps),
                                np.minimum(np.full(y_pred.shape, 1 - self.eps), y_pred))
            # likelihood function
            likelihood = np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred))
            loss = (Y - y_pred)  # + (0.00003*np.sum(self.w*self.w))
            # derivative of likelihood function
            dl = np.average(loss * X.T, axis=1)
            # updating the W parameter
            self.w = self.w + self.eta * dl
            # updating the bias parameter
            self.b = self.b + self.eta * np.average(loss)
            loss_summary.append(likelihood)
            if i % 1000 == 0:
                print("Iter", i, ":", likelihood)
            if len(loss_summary) >= 2 and abs(loss_summary[-2] - loss_summary[-1]) < self.eps:
                break
        return loss_summary

    def predict(self, test_x, threshold=0.5):
        """
        Predict thee testing data with weights W and bias b and sigmoid

        :param test_x: Testin data's input values. Here array of (2000,2)
        :param threshold: Threshold to classify the image in either class. Default is 0.5
        :return: Array of predicted class with respect to input. Here array of (2000)
        """
        y_pred = sigmoid(np.dot(self.w, test_x.T) + self.b)
        # thresholding the y_pred to extract accuracy
        y_pred[np.where(y_pred >= threshold)] = 1
        y_pred[np.where(y_pred < threshold)] = 0
        return y_pred


class GaussianNB:
    def __init__(self, eps=1e-7):
        """
        Setting up attributes for Naive Bayes.
        prior: Array of priors P(Y)
        n_classes: number of classes extracted from Training outputs.
        class_count: examples per class to calculate priors.
        mu: average of examples per class per feature
        sigma: variance of examples per class per feature
        :param eps: To add it to sigma in case sigma goes to zero and division error occurs.
        """
        self.prior = None
        self.eps = eps
        self.n_classes = 0
        self.class_count = None
        self.mu = None
        self.sigma = None

    def stats_from_params(self, x, y):
        """
        Calculating per class per feature mu, sigma and
        per class prior, example count.
        :param x: Training input data. Here array of (12000,2)
        :param y: Training output data. Here array of (12000)
        :return:
        """
        for c in range(self.n_classes):
            x_ind = x[y == c, :]
            self.class_count[c] = x_ind.shape[0]
            self.mu[c] = np.average(x_ind, axis=0)
            self.sigma[c] = np.var(x_ind, axis=0)
        self.prior = self.class_count / self.class_count.sum()
        self.sigma[:, :] += self.eps

    def weight_init(self, n_feature):
        """
        Setting empty boxes for class_count, mu, sigma, prior
        :param n_feature: Number of features in input data. Here it is 2.
        :return:
        """
        self.class_count = np.zeros(self.n_classes, dtype=np.float64)
        self.mu = np.zeros((self.n_classes, n_feature), dtype=np.float64)
        self.sigma = np.zeros((self.n_classes, n_feature), dtype=np.float64)
        self.prior = np.zeros(self.n_classes, dtype=np.float64)

    def fit(self, x, y):
        """
        Initializing the paramters of NB and calculating it's values.
        :param x: Training input data. Here array of (12000,2)
        :param y: Training output data. Here array of (12000)
        :return:
        """
        self.n_classes = np.unique(y).shape[0]
        n_feature = x.shape[1]
        # creating empty boxes for Mu, Sigma, Prior and Class count
        self.weight_init(n_feature)
        # calculating the Mu, Sigma, Prior and Class count from traninig input and output
        self.stats_from_params(x, y)

    def get_gaussian_estimation(self, tsX):
        """
        predicting the testing input using NB paramters calculated before.
        :param tsX: Testing input data. Here array of (2000, 2)
        :return:
        """
        gaussian_estimation = []
        for i in range(self.n_classes):
            # the log of prior log(P(y))
            prior = np.log(self.prior[i])
            # the sum of normal distribution estimation for each feature
            n_xy = -0.5 * np.sum(np.log(2. * np.pi * (self.sigma[i])))
            n_xy -= 0.5 * np.sum(((tsX - self.mu[i]) ** 2) / (self.sigma[i]), 1)
            # propability P(Y|X)
            gaussian_estimation.append(prior + n_xy)
        gaussian_estimation = np.array(gaussian_estimation).T
        return gaussian_estimation

    def predict(self, x):
        """
        From testing data predicting the gaussian estimation and output class.
        :param x: Testing input data. Here array of (2000, 2)
        :return:
        """
        e = self.get_gaussian_estimation(x)
        # argmax to find the predicted class
        return np.argmax(e, axis=1)


def sigmoid(features):
    """
    sigmoid function
    :param features: input features for whom the sigmoid is calculated.
    :return: return the output of sigmoid function. ranges from [0,1]
    """
    return 1 / (1 + np.exp(-features))


def find_features(trX):
    """
    From input vector finding 2 parameters per input raw. Mean, Standard Deviation
    :param trX: Training input data. Here array of (12000,784)
    :return: (12000,2) array by putting together 2 paramters.
    """
    meanX = np.average(trX, axis=1)
    stdX = np.std(trX, axis=1)
    # stacking 2 axes vertically to make (12000, 2)
    return np.vstack((meanX, stdX)).T


def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['trX'], np.reshape(data['trY'], -1), data['tsX'], np.reshape(data['tsY'], -1)


def normalize_data(trainX, testX):
    """
    Normalizsing the input by subtracting meand and dividing by standard deviation.
    Here mean and standard deviation are caluclated only for traning data.
    The testing data is normalized by same training mean and standard deviation.
    :param trainX: Training input data. Here (12000, 2)
    :param testX: Testing input data. Here (2000, 2)
    :return: Normalized output vectors
    """
    mtrX, strX = np.mean(trainX), np.std(testX)
    # x = (x-mu)/std    Will use same mu and std from training data for test data as well.
    return (trainX - mtrX) / strX, (testX - mtrX) / strX


def prepare_input(file_path, norm=True):
    """
    Loading data, finding 2 features per raw and normalizing data
    :param file_path:
    :param norm: True if you want to normalize data else False
    :return: training and testing, input and output
    """
    # reading data from file
    train_x, train_y, test_x, test_y = load_data(file_path)
    # finding per image mean and standard deviation
    train_x, test_x = find_features(train_x), find_features(test_x)
    # Normalizing data if flag id true
    if norm:
        train_x, test_x = normalize_data(train_x, test_x)
    return train_x, test_x, train_y, test_y


def confusion_matrix(y_real, y_pred):
    """
    Calculating confusion matrix for predicted vs actual output
    :param y_real: Actual output
    :param y_pred: Predicted output
    :return: confusion matrix
    """
    # Distinct y values in y_real
    unique = np.unique(y_real)
    n = len(unique)
    # confusion matrix empty box
    confusion_mtr = np.zeros((n, n))
    # We actually don't need this but using it as it will run for other problems
    label_to_ind = {y: x for x, y in enumerate(unique)}
    for i in range(len(y_real)):
        x = label_to_ind[y_real[i]]
        y = label_to_ind[y_pred[i]]
        confusion_mtr[x][y] += 1
    return confusion_mtr


def classwise_accuracy(y_real, y_pred):
    """
    Calculating class-wise accuracy using confusion matrix
    :param y_real: Actual output
    :param y_pred: Predicted output
    :return: per class accuarcy
    """
    #  Divide the matrix raw by sum of values in that raw
    cm = confusion_matrix(np.reshape(y_real, -1), np.reshape(y_pred, -1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # now the diagonal will be our classwise accuracy numbers
    return cm.diagonal()


st = time.time()
path = "fashion_mnist.mat"
# If you will set norm flag true, It will normalize the input before sending it to models.
ftrX, ftsX, trY, tsY = prepare_input(path, norm=False)

# Model Creation
logr = LogisticReg()
summary = logr.train(ftrX, trY, 50000)
y_hat = logr.predict(ftsX)

# Accuracy calculation
print("Accuracy for Logistic regression", classwise_accuracy(tsY, y_hat))

# Model Creation
g = GaussianNB()
g.fit(ftrX, trY)
estimation = g.predict(ftsX)

# Accuracy calculation
print("Accuracy for Naive bayes", classwise_accuracy(tsY, estimation))

print("Total time taken to run 2 models", time.time() - st)
