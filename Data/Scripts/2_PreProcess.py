# Import required libraries
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame
from scipy.spatial import distance
from matplotlib import rc
from plot_config import *

pd.options.mode.chained_assignment = None  # default='warn'

# Load ./dfData.parquet
# sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")
dfData.loc[dfData['zip'].astype(str).str.len() > 4, 'zip'] = np.nan
dfData.loc[dfData['customer_zip'].astype(str).str.len() > 4, 'customer_zip'] = np.nan


# Split data into wip and finished jobs. This serves as test and train data.
dfDataFinished = dfData[dfData['wip'] == 0]
dfDataWIP = dfData[dfData['wip'] == 1]

# Number of unique finished jobs and WIP jobs
nFinished = len(dfDataFinished['job_no'].unique())
nWIP = len(dfDataWIP['job_no'].unique())
print(f"The finished jobs account for {round(nFinished / (nFinished + nWIP) * 100, 2)}% of the total number of jobs.")

# Average, Max and Min number of observations per finished job
obs = dfDataFinished.groupby('job_no', observed=True).size()
print(f"The average number of observations per finished job is {round(obs.mean(), 2)}.")
print(f"The max number of observations per finished job is {obs.max()}.")
print(f"The min number of observations per finished job is {obs.min()}.")

# Fraction of finished jobs with less than 4 observations
fraction = (dfDataFinished.groupby('job_no', observed=True).filter(lambda x: len(x) < 4)[
                'job_no'].nunique() / nFinished) * 100
print(f"The fraction of finished jobs with less than 4 observations is {round(fraction, 2)}%.")

# Omit finished jobs with less than 4 observations
dfDataFinished = dfDataFinished.groupby('job_no').filter(lambda x: len(x) >= 4)

# Split the finished jobs into train and test (Assuming 'train' column exists and is binary)
dfDataFinishedTrain = dfDataFinished[dfDataFinished['train'] == 1]
dfDataFinishedTest = dfDataFinished[~dfDataFinished.index.isin(dfDataFinishedTrain.index)]


# Feature scaling function
def scale_data(train, test, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    train_scaled = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)
    return train_scaled, test_scaled, scaler


# Scale all numeric columns
numeric_cols = dfDataFinishedTrain.select_dtypes(include=[np.number]).columns.tolist()
train_data = dfDataFinishedTrain[numeric_cols]
test_data = dfDataFinishedTest[numeric_cols]

# Check for infinite values in the training and test data
infinite_train_vals = np.isinf(train_data).sum()
infinite_test_vals = np.isinf(test_data).sum()

# Filter columns with infinite values
infinite_train_cols = infinite_train_vals[infinite_train_vals > 0].index.tolist()
infinite_test_cols = infinite_test_vals[infinite_test_vals > 0].index.tolist()

# Replace infinite values with NaN
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaN values with the mean of the column
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Scale the data
scaled_train, scaled_test, scaler = scale_data(train_data, test_data)

# To invert scaling
inverted_train = pd.DataFrame(scaler.inverse_transform(scaled_train), columns=train_data.columns)

# Split into dependent and independent variables
sDepVar = 'total_contribution'
colIndepVar = [col for col in dfDataFinished.columns if col != sDepVar]

dfDataFinishedTrainIndep = dfDataFinishedTrain[colIndepVar]
dfDataFinishedTestIndep = dfDataFinishedTest[colIndepVar]

dfDataFinishedTrainDep = dfDataFinishedTrain[['date', 'job_no', sDepVar]]
dfDataFinishedTestDep = dfDataFinishedTest[['date', 'job_no', sDepVar]]

### Predict sDepVar using OLS ###
# Import required libraries
import statsmodels.api as sm

# Get the 5 most correlated variables (of numeric variables)
corr = dfDataFinishedTrain[numeric_cols].corr()
corr = corr.sort_values(by=sDepVar, ascending=False)
corr = corr[sDepVar]
# Filter out variables with "contribution" or "revenue" in the name
corr = corr[~corr.index.str.contains('contribution|revenue')]
corr = corr[1:6]
# Save the 5 most correlated variables in a list
lIndepVar = corr.index.tolist()

# Run OLS
model = sm.OLS(dfDataFinishedTrain[sDepVar], dfDataFinishedTrain[lIndepVar])
results = model.fit()
print(results.summary())
# Save results to LaTeX
ols = results.summary().as_latex()
with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)
