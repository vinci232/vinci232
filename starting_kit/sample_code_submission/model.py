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
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score

from data_manager import DataManager

import matplotlib.pyplot as plt


class Preprocessing(BaseEstimator):

    def __init__(self):
        lil_clf = SVC(kernel='linear') # classifieur lineaire
        self.transformer = PCA(n_components=100) # on veut que le resultat soit compose de 100 features
        """
        self.pipe = Pipeline(BaseEstimator)
        self.pipe 
        Pipeline(memory=None, steps=[('reduction_dim', self.fit(self, data.data['Xtrain'], data.data['Ytrain'])), ('lil_clf', lil_clf)])
        Pipeline(steps=[('process', Preprocessing()), ('clf', self.classifier)])
        """
        
    def fit(self, Xtrain, Ytrain):
        # premiere methode de preprocessing
        X_scaled = preprocessing.scale(X_train)
        Y_scaled = preprocessing.scale(Y_train)
        Xtrain_transf=self.transformer.fit_transform(Xtrain)
        Ytrain_transf=self.transformer.fit_transform(Ytrain)       
        return Xtrain_transf,Ytrain_transf

    def fit_transform(self, X, Y):
        return self.transformer.fit_transform(X,Y)

    def transform(self, X, Y):
        return self.transformer.transform(X,Y)

class model (BaseEstimator):

    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation. 
        '''
        self.num_train_samples = 65856
        self.num_feat = 100
        self.num_labels = 2
        self.is_trained = False
        self.model = neural_network.MLPClassifier(activation = 'relu', solver = 'adam', learning_rate = 'adaptive', learning_rate_init = 0.00005, beta_1 = 0.9, beta_2 = 0.999, max_iter = 500, epsilon = 1e-9)
        
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
        
        self.model.fit(X, y)
        print(self.model.score(X, y))

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
        
        return self.model.predict(X)
        

    def save(self, path="./"):
        pickle.dump(self.model, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
'''
def main() :
    
    This function is solely used in order to choose a good model,
    and to fine-tune the hyper-parameters.
    Hence, we could call it the 'test' function.
    This function is only called when ran by the Python interpretor,
    but if it's imported as a module, it is not ran unless we call it.
    
    
    M = model()
    P = Preprocessing()
    D = DataManager('perso', 'public_data', replace_missing = True)
    
    # Code for testing different features number values for the PCA
    
    features_number = []
    pca_scores = []
    for i in range(0, 210, 10):
        X_train = D.data['X_train']
        Y_train = D.data['Y_train']
        
        P.transformer = PCA(n_components = i)
        X_train = P.fit_transform(X_train, Y_train)
        M.model.fit(X_train, Y_train)
        
        temp_score = M.model.score(X_train, Y_train)
        features_number.append(i)
        
        pca_scores.append(temp_score)
        print("features = ", i, "; score = ", temp_score)
    
    plt.plot(features_number, pca_scores, 'r+')
    plt.show()
    
    # Code for testing different classifiers.
    # We will keep the classifier with the highest score, with their default hyperparameters
    
    X_train = D.data['X_train']
    Y_train = D.data['Y_train']
    
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
    
    # This block is commented after its first execution.
    # Indeed, we only need it once in order to give us the best classifier to use.
    # But since it has to build a whole model and fit it at each iteration, it is really long to use.
    # We first use in order to get the best classifier, then we comment it in order to gain in efficiency.
    for i in range(len(classifiers)):
        M.model = classifiers[i]
        M.model.fit(X_train, Y_train)
        scores[classifier_names[i]] = M.model.score(X_train, Y_train)
        print(scores)
    
    # Here we can see that many different classifiers give a perfect score of 1.0
    # Some always give bad results, GaussianNB for example is one of them
    # And some always give a very good result (decision_tree, gaussian_process and neural_network)
    # If we use the jupyter notebook in order to give precise metrics for our classifiers, we can see that the most effective seems to be neural_network.
    # We will then use neural_network as our classifier because it also presents a very good cross validation score.
    
    # Now, let's fine tune the hyperparameters of this classifier
    # We refer you to the classifier's documentation:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # We are going to use a dichtomy algorithm on specific hyperparameters that we selected
    
    hyperparameters = {
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'learning_rate' : ['constant', 'invscaling', 'adaptative'],
    'learning_rate_init' : 'float',
    'beta_1' : 'float',
    'beta_2' : 'float',
    'max_iter' : 'int'}
    
    # Same as before, after the first run, we took the best hyperparameters and commented the whole thing.
    # We keep this code block here for reference, and for you to understand how we fine-tuned our HPs.
    for i in range(len(hyperparameters['activation'])):
        M.model = neural_network.MLPClassifier(activation = hyperparameters['activation'][i])
        M.model.fit(X_train, Y_train)
        print("activation " + hyperparameters['activation'][i] + " : " + M.model.score(X_train, Y_train))
    
    for i in range(len(hyperparameters['solver'])):
        M.model = neural_network.MLPClassifier(solver = hyperparameters['solver'][i])
        M.model.fit(X_train, Y_train)
        print("solver " + hyperparameters['solver'][i] + " : " + M.model.score(X_train, Y_train))
    
    for i in range(len(hyperparameters['learning_rate'])):
        M.model = neural_network.MLPClassifier(learning_rate = hyperparameters['learning_rate'][i])
        M.model.fit(X_train, Y_train)
        print("learning_rate " + hyperparameters['learning_rate'][i] + " : " + M.model.score(X_train, Y_train))
    
    # Since these algorithms are really long to run, we only have time to make 5 iterations.
    for i in range(5):
        m = 0.0001
        M = 0.001
        s = 0.9657
        S = 0.8949
        # These HP values were ran by hand to initialize the dichotomy
        M.model = neural_network.MLPClassifier(learning_rate_init = (m+M)/2)
        s1 = M.model.score(X_train, Y_train)
        if (s1 > s) :
            m = (m+M)/2
            s = s1
            best_value = m
        else :
            M = (m+M)/2
            S = s1
            best_value = M
    print("learning_rate_init : " + best_value)
    
    for i in range(5):
        m = 0.8
        M = 0.9
        s = 0.9598
        S = 0.9657
        # These HP values were ran by hand to initialize the dichotomy

        M.model = neural_network.MLPClassifier(beta_1 = (m+M)/2)
        s1 = M.model.score(X_train, Y_train)
        if (s1 > s) :
            m = (m+M)/2
            s = s1
            best_value = m
        else :
            M = (m+M)/2
            S = s1
            best_value = M
    print("beta_1 : " + best_value)
    
    # beta_2 fine-tuning took too long to run. We had to kill the process.
    for i in range(5):
        m = 0.0001
        M = 0.001
        s = 0.9657
        S = 0.8949
        # These HP values were ran by hand to initialize the dichotomy
        M.model = neural_network.MLPClassifier(beta_2 = (m+M)/2)
        s1 = M.model.score(X_train, Y_train)
        if (s1 > s) :
            m = (m+M)/2
            s = s1
        else :
            M = (m+M)/2
            S = s1
'''