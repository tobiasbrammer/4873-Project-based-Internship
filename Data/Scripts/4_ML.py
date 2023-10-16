# Import required libraries
import os
import runpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
import joblib
from pandas import DataFrame
from scipy.spatial import distance
from matplotlib import rc
from plot_config import *
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

# Load ./dfData.parquet
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
# sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)

# Load ./dfDataTest.parquet and ./dfDataTrain.parquet
dfDataTest = pd.read_parquet("./dfDataTest.parquet")
dfDataTrain = pd.read_parquet("./dfDataTrain.parquet")

# Import scales
x_scaler = joblib.load("./.AUX/x_scaler.save")
y_scaler = joblib.load("./.AUX/y_scaler.save")


### Elastic Net Regression ###

# Import required libraries
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

# Define X and y
X = dfDataFinishedTrain[numeric_cols]
y = dfDataFinishedTrain['final_estimate_costs']
X_test = dfDataFinishedTest[numeric_cols]
y_test = dfDataFinishedTest['final_estimate_costs']


# Define Elastic Net model
elastic_net = ElasticNet()

# Define hyperparameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio': [0.001, 0.01, 0.1, 1, 10, 100]}
elastic_net_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit to the training data
elastic_net_cv.fit(X, y)

# Predict on the test data


