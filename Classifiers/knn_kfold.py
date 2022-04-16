
from cmath import sqrt
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

features, labels = load_wine(return_X_y = True)

class_0 = 0
class_1 = 1

features = features[(labels == class_0) | (labels == class_1)]
labels = labels[(labels == class_0) | (labels == class_1)]

num_data, num_features = features.shape

features, labels = shuffle(features, labels, random_state=1)


# This holds the average error rate on the test folds for each value of k
k_error_rates = []

for k in [23, 51, 101]:

    k_fold = KFold(n_splits=5)

     # This holds the error rates on each of the 5 test folds for a specific value of k
    error_rates = []
    for train_idx, test_idx in k_fold.split(features):

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        classifications = []

        # iterate through each test data point to calculate predicted class
        for test_idx, test_feature in enumerate(test_features):
            euc_dists = []
            # calculate euc distance from selected test_feature to all training features.
            for train_feature in train_features:
                delta = (test_feature - train_feature)
                sq_delta = np.dot(delta.T, delta)
                euc_dist = np.sqrt(sq_delta)
                euc_dists.append(euc_dist)
                # euc_dist_check = np.linalg.norm(test_feature - train_feature)
            # merge with labels
            labeled_distances = np.concatenate((euc_dists, train_labels), axis = 0)
            # reshape to matrix
            labeled_distances = labeled_distances.reshape(2,104)
            # sort by nearest distance (distance row only)
            labeled_distances = labeled_distances[:, labeled_distances[0, :].argsort()]
            # transpose for columns of distance, class
            labeled_distances = labeled_distances.T
            # select first k rows 
            k_distances = labeled_distances[:k,]
            # predict test_feature class from closest k training_features
            pred_class = round(np.mean(k_distances[:,1]))
            # add to array tracking classification predictions for each fold
            classifications.append(pred_class)
        # error for each fold
        classification_errors = classifications - test_labels
        classification_error_rate =  1 - ((len(classification_errors) - np.count_nonzero(classification_errors))/len(classification_errors))
        # append each fold error to k-level error calculation
        error_rates.append(classification_error_rate)
    k_error_rates.append(np.average(error_rates))

print('Average test error for each value of k:', k_error_rates)
