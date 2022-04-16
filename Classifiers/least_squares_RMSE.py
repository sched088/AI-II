
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_linnerud
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import numpy as np

features, target_vals = load_diabetes(return_X_y = True)

num_data, num_features = features.shape

# Append a value of 1 to each data point feature vector so we fit the intercept and increment num features
features = np.insert(features, num_features, 1, axis=1)
num_features += 1 

features, target_vals = shuffle(features, target_vals, random_state=1)


# This holds the average error rate on the test folds for each value of lambda
lambda_val_rmse = []

for lambda_val in [0.1, 1, 10, 100]:

    k_fold = KFold(n_splits=5)

    # This holds the error rates on each of the 5 test folds for a specific value of k
    rmse_vals = []

    for train_idx, test_idx in k_fold.split(features):

        train_features = features[train_idx]
        train_target_vals = target_vals[train_idx]
        test_features = features[test_idx]
        test_target_vals = target_vals[test_idx]

        # set variables to relevant names
        identity_matrix_i = np.identity(len(train_features[0,:]))
        design_matrix_x = train_features
        target_value_vector_y = train_target_vals

        # perform regularized linear regression on whole matrix
        xty = (np.transpose(design_matrix_x).dot(target_value_vector_y))
        xtx = (np.transpose(design_matrix_x).dot(design_matrix_x))
        lambda_i = lambda_val*identity_matrix_i
        xtx_lambda_i = xtx + lambda_i
        neg_power_xtx_lambda_i = np.linalg.inv(xtx_lambda_i)
        w_star = np.transpose(neg_power_xtx_lambda_i.dot(xty))

        # calculate RMSE
        wtx_test = np.transpose(w_star).dot(np.transpose(test_features))
        y_sub_wtx_test = test_target_vals - wtx_test
        y_sub_wtx_test_sqr = np.power(y_sub_wtx_test, 2)
        sum_y_sub_wtx_test_sqr = sum(y_sub_wtx_test_sqr)
        rmse = np.sqrt((1/len(test_target_vals))*sum_y_sub_wtx_test_sqr)
        rmse_vals.append(rmse)

    lambda_val_rmse.append(np.average(rmse_vals))

print('Average test RMSE for each value of lambda:', lambda_val_rmse)
