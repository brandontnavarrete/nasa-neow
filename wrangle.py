import pandas as pd
import numpy as np
import os
import math


import matplotlib.pyplot as plt

import seaborn as sns
from seaborn import heatmap
import scipy.stats as stats
from scipy.stats import mannwhitneyu
import statistics

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# data split
from sklearn.model_selection import train_test_split

# sklearn metrics
from sklearn.metrics import accuracy_score, precision_score
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Logistic Regression
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------
'''
returns a data frame from a locally stored csv, returns a dataframe and prints the shape of that dataframe.
'''

def get_data():
    # reads csv 
    df = pd.read_csv('neo_v2.csv')
    
    # print shape
    print(df.shape)
    
    return df
# --------------------------------

def clean_nasa(df):
    
    ''' 
    Input : dataframe
    Output: cleaned dataframe
    
    function takes in dataframe and drops columns that are not helpful and returns the new dataframe
    ''' 
    
    # drop, they all orbit earth
    df = df.drop(columns = 'orbiting_body')
    
    # drop, these are all exlcuded from the sentry automated collison monitoring system
    df = df.drop(columns = 'sentry_object')
    
    
    df = df.drop(columns = 'id')
    df = df.drop(columns = 'name')
    
    return df



# -split-data-------------------------------

def split_data(df,strat):
    
    ''' 
    Input: Dataframe, Stratify 'target'
    Output: train, validate, test dataframes
    
    splits data frame and returns a train, validate, 
    and test data frame stratified on churn 
    
    This function will take one dataframe and a target to be stratified and return 3 split data frames(train,validate,test)
    '''
    
    # split df into train_validate and test
    train_validate, test = train_test_split(df,test_size =.2, 
                                             random_state = 42,
                                             stratify = df[strat])
    
    # split train_validate into train and validate
    train, validate = train_test_split(train_validate,
                                      test_size = .3,
                                      random_state = 42,
                                      stratify = train_validate[strat])
                                            
    # reset index for train validate and tes
    train.reset_index(drop=True, inplace=True)
    validate.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    
    print (train.shape,test.shape,validate.shape)
    return train, validate, test
    
# -x-and-y-----------------------------
    
def x_and_y(train,validate,test,target):
    
    """
    splits train, validate, and target into x and y versions
    """

    x_train = train.drop(columns= target)
    y_train = train[target]

    x_validate = validate.drop(columns= target)
    y_validate = validate[target]

    x_test = test.drop(columns= target)
    y_test = test[target]
    
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    y_test = pd.DataFrame(y_test)



    return x_train,y_train,x_validate,y_validate,x_test, y_test
    
    