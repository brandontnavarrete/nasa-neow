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

a = .05

#-----------------------------------

def how_many(df):
    
    # separating groups
    haz = df[df.hazardous == True]
    safe = df[df.hazardous == False]
    
    # finding percentage
    len(haz) / len(safe)
    
    sns.countplot(data = safe, x = 'hazardous',alpha = .5,label = 'safe',color = 'grey')
    sns.countplot(data = haz, x = 'hazardous',label = 'hazardous',color = 'red')
    plt.title('10 % of the Objects are Dangerous')
    plt.xlabel('Safety Label')
    plt.ylabel('Frequency')
    plt.legend()
    None
    
# question 2 functions------------------

def plt_diam(df):
    
    sns.set_style('darkgrid')
    sns.relplot(data = df, x = 'est_diameter_max',y= 'hazardous',hue = "hazardous",palette = "ocean_r")
    plt.title('Does Diameter tell us if an Asteroid is Hazardous')
    plt.ylabel('Danger Status')
    plt.xlabel('Estimated Minimum Diameter')
    None


# ---------------------------------------


def diam_stats(df):

        haz = df[df.hazardous == True]
        safe = df[df.hazardous == False]
        
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
    
    sns.set_style('darkgrid')
    sns.relplot(data = df, x = 'absolute_magnitude',y= 'hazardous',hue = "hazardous",palette = "ocean_r")
    plt.title('Does Magnitude tell us if an Asteroid is Hazardous')
    plt.ylabel('Danger Status')
    plt.xlabel('Visbility Rating')
    None


# ---------------------------------------


def mag_stats(df):

        haz = df[df.hazardous == True]
        safe = df[df.hazardous == False]
        
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
        

    
