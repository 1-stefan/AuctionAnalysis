import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# for data manipulation
import pandas as pd, numpy as np, random

#for graphs
import matplotlib.pyplot as plt
import seaborn as sns

#models to run
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn import neighbors
from sklearn import tree

#train_test_split
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit, GridSearchCV

#for cycle
from itertools import cycle

#metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVR, SVR

RANDOM_STATE = 42
random.seed(RANDOM_STATE)

import math
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

class LR_reg:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        """
        Initialize the KNN classifier.

        Parameters:
        -----------
            X (pandas.DataFrame): The feature matrix of shape (n_samples, n_features).
            y (pandas.Series): The target vector of shape (n_samples,).
            n_neighbors (int, optional): The number of nearest neighbors to use in classification. Defaults to 5.
            test_size (float, optional): The proportion of samples to use for testing. Defaults to 0.2.
            random_state (int, optional): The random state to use for splitting the data. Defaults to 42.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.model = None
        self.best_params = None
        self.best_score = None
    def fit(self, tune_fit="no"):
        self.model = LinearRegression().fit(self.X_train, self.y_train)
    def predict(self):
        #predict Price based on test set
        return self.model.predict(self.X_test)
    def score(self):
        """
        Calculate the accuracy score and classification report for the KNN model.

        Parameters:
        -----------
        None

        Returns:
        -----------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class.
        """
        y_pred = self.predict()
        #score1 = self.r2_score(self.X_test, self.y_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = MAE(self.y_test, y_pred)
        mse = MSE(self.y_test, y_pred)
        return  r2, mae, mse
    

class KNN_reg:
    def __init__(self, X, y, n_neighbors=5, test_size=0.2, random_state=42):
        """
        Initialize the KNN classifier.

        Parameters:
        -----------
            X (pandas.DataFrame): The feature matrix of shape (n_samples, n_features).
            y (pandas.Series): The target vector of shape (n_samples,).
            n_neighbors (int, optional): The number of nearest neighbors to use in classification. Defaults to 5.
            test_size (float, optional): The proportion of samples to use for testing. Defaults to 0.2.
            random_state (int, optional): The random state to use for splitting the data. Defaults to 42.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.n_neighbors = n_neighbors
        self.model = None
        self.best_params = None
        self.best_score = None
        
    def fit(self, tune_fit="no"):
        """
        Fit the KNN model to the training data.

        Parameters:
        -----------
            tune_fit (str, optional): Whether to perform hyperparameter tuning. If "yes", performs a grid search to find
                the best hyperparameters. Defaults to "no".

        Raises:
        -----------
            ValueError: If `tune_fit` is not "yes" or "no".

        Returns:
        -----------
            None
        """
        if tune_fit=="yes": 
            estimator_KNN = KNeighborsRegressor(algorithm='auto')
            parameters_KNN = {
                'n_neighbors': (1,31,1),
                'leaf_size': (1,50,1),
                'p': (1,2),
                'weights': ('uniform', 'distance'),
                'metric': ('manhattan', 'euclidean')
            }
                   
                # with GridSearch
            grid_search = GridSearchCV(
                estimator=estimator_KNN,
                param_grid=parameters_KNN,
                scoring = 'neg_mean_squared_error',
                n_jobs = -1,
                cv = 5
            )
            
            
            self.model = grid_search.fit(self.X_train, self.y_train)
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
        elif tune_fit=="no":
             self.model = KNeighborsRegressor(n_neighbors=5).fit(self.X_train, self.y_train)
        else:
            raise ValueError("Invalid value for `tune_fit`. Must be either 'yes' or 'no'.")
        
    def predict(self):
        """
        Predict the target values for the test data.

        Parameters:
        -----------
        None

        Returns:
        -----------
            numpy.ndarray: The predicted target values of shape (n_samples,)
        """
        return self.model.predict(self.X_test)
    
    def score(self):
        """
        Calculate the accuracy score and classification report for the KNN model.

        Parameters:
        -----------
        None

        Returns:
        -----------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class.
        """
        y_pred = self.predict()
        r2 = r2_score(self.y_test, y_pred)
        mae = MAE(self.y_test, y_pred)
        mse = MSE(self.y_test, y_pred)
        return  r2, mae, mse


class RandomForest_reg:
    def __init__(self, X, y, random_state=42):
        """Initializes the RandomForest class with the input features X and target variable y,
        and the random state used for reproducibility.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            The target variable of shape (n_samples,).
        random_state : int, default=RANDOM_STATE, which is 42
            The seed value for random number generator used to split the data.

        Returns:
        --------
        None"""
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.best_score = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    def fit(self, tune_fit="yes"):
        """Trains the random forest model using the input data X and y.
        If tune_fit is set to "yes", it tunes the hyperparameters using GridSearchCV(), otherwise, with default parameters.
        It then stores the best hyperparameters and best estimator in the attributes best_params and model, respectively.

        Parameters:
        -----------
        tune_fit : str, default="yes"
            If "yes", tune the hyperparameters using GridSearchCV(), otherwise use default parameters.

        Returns:
        --------
        None
        """

        if tune_fit=="yes":
            param_grid = [{'n_estimators': [50, 100, 150],
                'max_depth': [10,25,50,100],
                'min_samples_leaf': [1, 2, 4]}]
            grid_search = GridSearchCV(
                RandomForestRegressor(random_state=self.random_state), 
                param_grid, 
                cv=5
            )
            self.model = grid_search.fit(self.X_train, self.y_train)
            self.best_params = grid_search.best_params_
            self.best_score = grid_search.best_score_
        else: 
            self.model = RandomForestRegressor(random_state=self.random_state).fit(self.X_train, self.y_train)
            
    def predict(self):
        """Predicts the target variable of the test data using the trained random forest model.
        Returns the predicted target variable values.

        Parameters:
        -----------
        None

        Returns:
        --------
        numpy.ndarray: The predicted target variable values
        
        """
        return self.model.predict(self.X_test)
    
    def score(self):
        """Computes the accuracy score and classification report for the predicted target variable and the actual test target variable.

        Parameters:
        -----------
        None

        Returns:
        --------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class."""
        y_pred = self.predict()
        r2 = r2_score(self.y_test, y_pred)
        mae = MAE(self.y_test, y_pred)
        mse = MSE(self.y_test, y_pred)
        return r2, mae, mse

class SVM_reg:
    def __init__(self, X, y, random_state=42):
        """Initializes the RandomForest class with the input features X and target variable y,
        and the random state used for reproducibility.

        Parameters:
        -----------
        X : pandas.DataFrame
            The input feature matrix of shape (n_samples, n_features).
        y : pandas.Series
            The target variable of shape (n_samples,).
        random_state : int, default=RANDOM_STATE, which is 42
            The seed value for random number generator used to split the data.

        Returns:
        --------
        None"""

        self.X = X
        self.y = y
        self.random_state = random_state
        self.model = None
        self.best_params = {} 
        self.best_score = 0 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=random_state)
    
    def fit(self, tune_fit="yes"):
        """Trains the random forest model using the input data X and y.
        If tune_fit is set to "yes", it tunes the hyperparameters using GridSearchCV(), otherwise, with default parameters.
        It then stores the best hyperparameters and best estimator in the attributes best_params and model, respectively.

        Parameters:
        -----------
        tune_fit : str, default="yes"
            If "yes", tune the hyperparameters using GridSearchCV(), otherwise use default parameters.

        Returns:
        --------
        None
        """
        if tune_fit=="yes":
            param_grid = {'C': [0.1, 1, 10, 100], 
              'gamma': [1, 0.1, 0.01 ],
              'kernel': ['linear']}
            grid_search = GridSearchCV(SVR(), param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
            self.model = grid_search.fit(self.X_train, self.y_train)
        else: 
            self.model = SVR(C=100, gamma=1, kernel='linear').fit(self.X_train, self.y_train)
            
    def predict(self):
        """Predicts the target variable of the test data using the trained random forest model.
        Returns the predicted target variable values.

        Parameters:
        -----------
        None

        Returns:
        --------
        numpy.ndarray: The predicted target variable values
        
        """
        return self.model.predict(self.X_test)
    
    def score(self):
        """Computes the accuracy score and classification report for the predicted target variable and the actual test target variable.

        Parameters:
        -----------
        None

        Returns:
        --------
        accuracy score: The ratio of the correctly predicted observations to the total observations.
        
        classification report: A text report of the main classification metrics such as precision, recall, f1-score and support for each class."""
        y_pred = self.predict()
        r2 = r2_score(self.y_test, y_pred)
        mae = MAE(self.y_test, y_pred)
        mse = MSE(self.y_test, y_pred)
        return  r2, mae, mse

'''
import os
cwd = os.getcwd()
print(cwd)
auction = pd.read_csv(f'{cwd}/data/processed/cleaned_auction.csv')
X = auction.drop(['price'], axis = 1)
y = auction['price']

fit_rf = RandomForest_reg(X, y, random_state=42)
fit_rf.fit(tune_fit = "yes")
pred_rf = fit_rf.predict()

print(fit_rf.score()[0])
'''