from sklearn.datasets import load_wine
from sklearn.utils import shuffle
import numpy as np

percent_train = 0.8

num_reps = 10

features, labels = load_wine(return_X_y = True)

class_0 = 0
class_1 = 1

features = features[(labels == class_0) | (labels == class_1)]
labels = labels[(labels == class_0) | (labels == class_1)]

num_data, num_features = features.shape
split = int(np.ceil(num_data*percent_train))

features, labels = shuffle(features, labels, random_state=1)

train_features, train_labels = (features[:split], labels[:split])
test_features, test_labels = (features[split:], labels[split:])

num_train_data = train_features.shape[0]
num_test_data = test_features.shape[0]


error_rates = []
num_features = train_features.shape[1]
iterations = 100000
prediction_vectors = []

for repetition in range(num_reps):
    print("repetition #: " + str(repetition))
    """
    # comment out matrix-wide appraoch
    if repetition == 0: 
        w = np.random.random([num_features,]) 
    alpha_t = 1 / np.sqrt(num_reps)
    wtx = np.transpose(w).dot(np.transpose(train_features))
    # apply indicator function (if positive then 1 if negative then 0)
    indfx_wtx = (wtx > 0).astype(int)
    w_t1 = np.transpose(w) + alpha_t*(train_labels - if_wtx).dot(train_features)
    w = np.transpose(w_t1)
    """
    for iteration in range(iterations):
        # status output
        # if iteration % 10000 == 0:
        # print(iteration)

        # initialize weights for first iteration only
        if iteration == 0:
            w = np.random.random([num_features,])
        # set learning rate
        alpha_t = 1 / np.sqrt(num_reps) 
        # randomly sample one piece of training data
        sample_id = np.random.randint(0,num_train_data-1)
        # define selected feature vector
        feature_vector = train_features[sample_id] 

        wtx = np.transpose(w).dot(np.transpose(feature_vector))

        # apply indicator function (if positive then 1 if negative then 0)
        indfx_wtx = (wtx > 0).astype(int)

        # update weight vector
        w_t1 = np.transpose(w) + alpha_t*(train_labels - indfx_wtx).dot(train_features)
        w = np.transpose(w_t1)
        prediction_vectors.append(w)

    # compute test set classification error (would normally use sklearn module...)
    mean_prediction_vectors = np.mean(prediction_vectors, axis = 0)
    wtx_test = mean_prediction_vectors.dot(np.transpose(test_features))
    indfx_wtx_test = (wtx_test > 0).astype(int)

    prediction_delta = test_labels - indfx_wtx_test
    error_rate = 1 - ((len(prediction_delta) - np.count_nonzero(prediction_delta))/len(prediction_delta))
    error_rates.append(error_rate)

print('Average error rate:', np.average(error_rates))
print('Error rate standard deviation:', np.std(error_rates))
