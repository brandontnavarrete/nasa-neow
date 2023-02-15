# imports
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
from sklearn.metrics import accuracy_score, recall_score
import sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Logistic Regression
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

a = .05

#-----------------------------------

def how_many(df):
    
    '''
    Input: dataframe
    Output: Chart
    Create separate dataframes based on target status and comparing their counts
    '''
    # separating groups
    haz = df[df.hazardous == True]
    safe = df[df.hazardous == False]
    
    # finding percentage
    len(haz) / len(safe)
    
    sns.countplot(data = safe, x = 'hazardous',alpha = .5,label = 'safe',color = 'grey')
    sns.countplot(data = haz, x = 'hazardous',label = 'hazardous',color = 'skyblue')
    plt.title('10 % of the Objects are Dangerous')
    plt.xlabel('Safety Label')
    plt.ylabel('Frequency')
    plt.legend()
    None
    
# question 2 functions------------------

def plt_diam(df):
    '''
    Input: dataframe
    Output: plot
    Creates a relplot and compares x and y with the hue set on "hazardous".
    
    '''
    # seaborn plot setting and titles
    # set grid style
    sns.set_style('darkgrid')
    # create relplot
    sns.relplot(data = df, x = 'est_diameter_max',y= 'hazardous',hue = "hazardous",palette = "ocean_r")
    plt.title('Does Diameter tell us if an Asteroid is Hazardous')
    plt.ylabel('Danger Status')
    plt.xlabel('Estimated Minimum Diameter')
    None


# ---------------------------------------


def diam_stats(df):
    haz = df[df.hazardous == True]
    
    
    
  
        # creating separate groups
    haz = df[df.hazardous == True]
    safe = df[df.hazardous == False]
        
        # creating series
    saf_min = safe.est_diameter_min
    haz_min = haz.est_diameter_min
        
        # mannwhitneyu do the distribution of t test
    t, p= mannwhitneyu(haz_min, saf_min)
                           
    print('Statistics=%.2f, p=%.2f' % (t, p))
                           # conclusion
    if p < a:
        print('Reject Null Hypothesis (Significant difference between two samples)')
    else:
        print('Do not Reject Null Hypothesis (No significant difference between two samples)')
                           
    
# question 3 functions------------------

def plt_mag(df):
    
   
    # seaborn plot setting and titles
    sns.set_style('darkgrid')
    sns.relplot(data = df, x = 'absolute_magnitude',y= 'hazardous',hue = "hazardous",palette = "ocean_r")
    plt.title('Does Magnitude tell us if an Asteroid is Hazardous')
    plt.ylabel('Danger Status')
    plt.xlabel('Visbility Rating')
    None


# ---------------------------------------


def mag_stats(df):
    

    # creating two groups 
    haz = df[df.hazardous == True]
    safe = df[df.hazardous == False]
        

        #creating a series
    saf_min = safe.absolute_magnitude
    
    haz_min = haz.absolute_magnitude
    

    # mannwhitneyu do the distribution of t test
    t, p= mannwhitneyu(haz_min, saf_min)
                           
    print('Statistics=%.2f, p=%.2f' % (t, p))

    # conclusion
    if p < a:
        print('Reject Null Hypothesis (Significant difference between two samples)')
    else:
        print('Do not Reject Null Hypothesis (No significant difference between two samples)')

# ---------------------------------------            
            
def plt_relv(df):
    
    '''
    Input: dataframe
    Output: plot
    Creates a relplot and compares x and y with the hue set on "hazardous".
    
    '''
    
    # seaborn plot setting and titles
    sns.set_style('darkgrid')
    sns.relplot(data = df, x = 'relative_velocity',y= 'hazardous',hue = "hazardous",palette = "ocean_r")
    plt.title('Does Velocity tell us if an Asteroid is Hazardous')
    plt.ylabel('Danger Status')
    plt.xlabel('Velocity')
    None


def relv_stats(df):
    
    
    # create two groups
    haz = df[df.hazardous == True]
    safe = df[df.hazardous == False]
        
    # create series
    saf_min = safe.relative_velocity
    haz_min = haz.relative_velocity
        
        # mannwhitneyu do the distribution of t test
    t, p= mannwhitneyu(haz_min, saf_min)
                           
    print('Statistics=%.2f, p=%.2f' % (t, p))
                           # conclusion
    if p < a:
        print('Reject Null Hypothesis (Significant difference between two samples)')
    else:
        print('Do not Reject Null Hypothesis (No significant difference between two samples)')

        
        
# ---------------------------------------
    
def baseline(y_train,y_validate,target):
    
    '''a function to return our baseline and create a dataframe to hold all the models and their features'''
    
    # turning our series into a data frame
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    
     # predict target value mode
    train_mode = y_train[target].mode()[0]
    validate_mode = y_validate[target].mode()[0]
   
    # predict tax value mode
    y_train[target + '_mode_baseline'] = train_mode
    y_validate[target + '_mode_baseline'] = validate_mode
    
    matching_values = (y_train['hazardous'] == y_train['hazardous_mode_baseline']).sum()

  # Calculate the baseline prediction
    baseline_pred = [statistics.mode(y_train['hazardous'])] * len(y_train['hazardous'])

    # Calculate the accuracy score
    baseline_accuracy = accuracy_score(y_train['hazardous'], baseline_pred)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}")

    # Calculate the recall score
    baseline_recall = recall_score(y_train['hazardous'], baseline_pred)
    print(f"Baseline Recall: {baseline_recall:.2f}")
    
    # creating a new series to hold our results of all model performance
    evals = {'Recall': [baseline_recall], 'model': ['baseline']}

    # creating a data frame from our series to pass on
    evals = pd.DataFrame(data=evals)
    
    return y_train,y_validate,evals,baseline_pred

# ---------------------------------------

def add_dt_test(x_train, y_train, x_test, y_test, md,evals):
    
    # create object instance

    clf = DecisionTreeClassifier(max_depth=md, random_state=42)
    
    # fit
    clf.fit(x_train, y_train['hazardous'])
    
    # Accuracy and Recall for the test data
    y_test_pred = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test['hazardous'], y_test_pred)
    test_recall = recall_score(y_test['hazardous'], y_test_pred)
    
    decision = {'Recall': test_recall, 'model': 'decision tree'}
    evals = evals.append(decision, ignore_index = True)
    
    return evals

# ---add_knn_test------------------------------------

def add_knn_test(x_train, y_train, x_validate, y_validate, x_test,y_test,evals):

    # create object instance
     knn = KNeighborsClassifier(n_neighbors = 40, weights= 'distance' )
    
    # fit
     knn.fit(x_train, y_train['hazardous'])
                                
      # Accuracy and Recall for the test data
     y_test_pred = knn.predict(x_test)
     test_accuracy = accuracy_score(y_test['hazardous'], y_test_pred)
     test_recall = recall_score(y_test['hazardous'], y_test_pred)    
                                
      # creating a new series to hold our results of all model performance
     decision = {'Recall': test_recall, 'model': 'knn'}
     evals = evals.append(decision, ignore_index = True)                       

     return evals 
    
# --add_xgboost-------------------------------------

def add_xgboost(x_train, y_train, x_validate, y_validate, x_test,y_test,evals):

    # create an instance with predetermined values 
    clf_xgb = xgb.XGBClassifier(objective ='binary:logistic', 
                                        seed = 42,
                                        max_depth = 3,    
                                        scale_pos_weight= 5,
                                        learning_rate = .01,
                                        subsample = .9,
                                        colsample_bytree = .5,
                                        n_jobs = 10)
    # fit model
    clf_xgb.fit(x_train,y_train['hazardous'],verbose = True)
    
    # predict values
    preds = clf_xgb.predict(x_test)
    test_accuracy = accuracy_score(y_test['hazardous'],preds)
    test_recall = recall_score(y_test['hazardous'],preds)
    
     # creating a new series to hold our results of all model performance
    decision = {'Recall': test_recall, 'model': 'XGBOOST'}
    evals = evals.append(decision, ignore_index = True)                       

    return evals, clf_xgb , preds

#----------------------------------------------------------

def plot_conf( y_test,preds,y_train,baseline_preds):
    
    '''
    Input: series
    Output: two confusion matrix
    The will plot a confusion matrix, side by side for comparison.
    '''
    fig, ax = plt.subplots(1,2)
    ax[0].set_title("XGBOOST Performance")
    ax[1].set_title("Baseline Performance")

    ConfusionMatrixDisplay(
    confusion_matrix = confusion_matrix(y_test, preds), 
    display_labels = ['Inert', 'Hazardous']).plot(ax=ax[0])

    ConfusionMatrixDisplay(
    confusion_matrix = confusion_matrix(y_train['hazardous'], baseline_preds), 
    display_labels = ['', '']).plot(ax=ax[1]);
