'''
Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
'''

import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import naive_bayes
from sklearn import gaussian_process
from sklearn import neural_network
from sklearn import svm
from sklearn import neighbors

class model (BaseEstimator):

    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples=65856
        self.num_feat=200
        self.num_labels=2
        self.is_trained=False
        self.model = clf = linear_model.LogisticRegression()
        
    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        
        '''
        This is the code given in example : 
        
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True
        self.model = self.model.fit(X, y)
        '''
        
        # Code for testing different classifiers.
        # We will keep the classifier with the highest score, with their default hyperparameters
        global clf
        
        scores = {}
        
        classifier_names = [
        "decision_tree",
        "logistic_regression",
        "random_forest",
        "naive_bayes",
        "gaussian_process",
        "neural_network",
        "svm",
        "radius_neighbors"]
        
        classifiers = [
        tree.DecisionTreeClassifier(),
        linear_model.LogisticRegression(),
        ensemble.RandomForestClassifier(),
        naive_bayes.GaussianNB(),
        gaussian_process.GaussianProcessClassifier(),
        neural_network.MLPClassifier(),
        svm.SVC(),
        neighbors.RadiusNeighborsClassifier()]
        
        """
        for i in range(len(classifiers)):
            clf = classifiers[i]
            clf.fit(X, y)
            scores[classifier_names[i]] = clf.score(X, y)

        print(scores)
        """
        
        # Here we can see that many different classifiers give a perfect score of 1.0
        # Some always give bad results, GaussianNB for example is one of them
        # And some always give a very good result (decision_tree, gaussian_process and neural_network)
        # If we use the jupyter notebook in order to give precise metrics for our classifiers, we can see that the most effective seems to be neural_network.
        # We will then use neural_network as our classifier because it also presents a very good cross validation score.
        
        # Now, let's fine tune the hyperparameters of this classifier
        # We refer you to the classifier's documentation:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        clf = neural_network.MLPClassifier(activation = 'relu', solver = 'adam', learning_rate = 'adaptive', learning_rate_init = 0.00005, beta_1 = 0.9, beta_2 = 0.999, max_iter = 500, epsilon = 1e-9)
        clf.fit(X, y)
        print(clf.score(X, y))

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        
        '''
        This again is the code given in the example :
        
        num_test_samples = X.shape[0]
        if X.ndim>1: num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = self.model.predict_proba(X)
        # If you uncomment the next line, you get pretty good results for the Iris data :-)
        #y = np.round(X[:,3])
        return y[:,1]
        '''
        
        global clf
        
        return clf.predict(X)
        

    def save(self, path="./"):
        pickle.dump(self.model, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
