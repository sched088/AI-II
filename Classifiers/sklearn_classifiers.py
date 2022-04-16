"""
> Logistic regression (set max iters = 3000),
> Linear support vector machine (SVM) (set kernel=‘linear’),
> Random forest (RF) (set criterion=‘entropy’, max depth=1),
> Adaboost

For each algorithm, use 5-fold cross validation (you may use the scikit-learn CV function6 for
this problem) to tune the following hyperparameters:

> Logistic regression7: C ∈[1e −5,1e −4,1e −3,1e −2,0.1,1,10,100,1000],
> SVM8: C ∈[1e −5,1e −4,1e −3,1e −2,0.1,1,10,100,1000],
> RF9: n estimators ∈[1,10,20,30,40,50,100,200],
> Adaboost10: n estimators ∈[1,10,20,30,40,50,100,200]
"""
#### DO NOT CHANGE THE BELOW CODE ####

from sklearn import svm
from sklearn.datasets import load_wine
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import numpy as np

features, labels = load_wine(return_X_y = True)

class_0 = 0
class_1 = 1

features = features[(labels == class_0) | (labels == class_1)]
labels = labels[(labels == class_0) | (labels == class_1)]

num_data, num_features = features.shape

features, labels = shuffle(features, labels, random_state=1)

def plot_it(y,std, x, classifier):
    import matplotlib.pyplot as plt
    import os

    cwd = os.getcwd()
    x = np.log(x)
    y = np.subtract(1,y)
    plt.errorbar(x, y, std, linestyle='None', marker = 'h')
    axis = plt.gca()
    axis.set_ylim([0,0.5])
    plt.savefig(cwd + classifier)
    plt.show()
    

def logistic_regression_classifier(features, labels):
    print('Running logistic_regression_classifier')
    mean_cv_scores = []
    std_cv_scores = []
    cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    max_iters = 3000

    for c in cs:
        # run classifier and output predictions
        classifier = LogisticRegression(C=c, max_iter=max_iters) # .fit(train_features, train_labels)
        # get classification scores
        cv_scores = cross_val_score(classifier, features, labels, cv = 5)
        # set scores for plotting
        mean_cv_scores.append(cv_scores.mean())
        std_cv_scores.append(cv_scores.std())

    plot_it(mean_cv_scores, std_cv_scores, cs, 'logistic_regression_classifier')


def support_vector_machine_classifier(features, labels):
    print('Running support_vector_machine_classifier')
    mean_cv_scores = []
    std_cv_scores = []
    cs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

    for c in cs:
        # set classifier
        classifier = svm.SVC(C=c, kernel='linear')
        # get classification scores
        cv_scores = cross_val_score(classifier, features, labels, cv = 5)
        # set scores for plotting
        mean_cv_scores.append(cv_scores.mean())
        std_cv_scores.append(cv_scores.std())

    plot_it(mean_cv_scores, std_cv_scores, cs, 'support_vector_machine_classifier')  


def random_forest_classifier(features, labels):
    print('Running random_forest_classifier')
    mean_cv_scores = []
    std_cv_scores = []
    n_estimators = [1, 10, 20, 30, 40, 50, 100, 200]
    for n_estimator in n_estimators:
        # set classifier
        classifier = RandomForestClassifier(n_estimators=n_estimator, criterion='entropy', max_depth=1)
        # get classification scores
        cv_scores = cross_val_score(classifier, features, labels, cv = 5)
        # set scores for plotting
        mean_cv_scores.append(cv_scores.mean())
        std_cv_scores.append(cv_scores.std())
    
    plot_it(mean_cv_scores, std_cv_scores, n_estimators, 'random_forest_classifier')  

def ada_boost_classifier(features, labels):
    print('Running ada_boost_classifier')
    mean_cv_scores = []
    std_cv_scores = []
    n_estimators = [1, 10, 20, 30, 40, 50, 100, 200]
    for n_estimator in n_estimators:
        # set classifier
        classifier = AdaBoostClassifier(n_estimators=n_estimator)
        # get classification scores
        cv_scores = cross_val_score(classifier, features, labels, cv = 5)
        # set scores for plotting
        mean_cv_scores.append(cv_scores.mean())
        std_cv_scores.append(cv_scores.std())
    
    plot_it(mean_cv_scores, std_cv_scores, n_estimators, 'ada_boost_classifier')  

logistic_regression_classifier(features, labels)
support_vector_machine_classifier(features, labels)
random_forest_classifier(features, labels)
ada_boost_classifier(features, labels)          


