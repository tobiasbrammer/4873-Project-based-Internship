# Import required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame
from scipy.spatial import distance
import os
from matplotlib import rc
from plot_config import *

# Load ./dfData.parquet
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(sDir)
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")

# Split data into wip and finished jobs
dfDataFinished = dfData[dfData['wip'] == 0]
dfDataWIP = dfData[dfData['wip'] == 1]

# Number of unique finished jobs and WIP jobs
nFinished = len(dfDataFinished['job_no'].unique())
nWIP = len(dfDataWIP['job_no'].unique())
print(f"The finished jobs account for {round(nFinished / (nFinished + nWIP) * 100, 2)}% of the total number of jobs.")

# Average, Max and Min number of observations per finished job
avg_obs = dfDataFinished.groupby('job_no').size().mean()
max_obs = dfDataFinished.groupby('job_no').size().max()
min_obs = dfDataFinished.groupby('job_no').size().min()
print(f"The average number of observations per finished job is {round(avg_obs, 2)}.")
print(f"The max number of observations per finished job is {max_obs}.")
print(f"The min number of observations per finished job is {min_obs}.")

# Fraction of finished jobs with less than 4 observations
fraction = (dfDataFinished.groupby('job_no').filter(lambda x: len(x) < 4)['job_no'].nunique() / nFinished) * 100
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


# Assuming you want to scale all numeric columns
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
# inverted_train = pd.DataFrame(scaler.inverse_transform(scaled_train), columns=train_data.columns)

# Split into dependent and independent variables
sDepVar = 'final_estimate_costs'
colIndepVar = [col for col in dfDataFinished.columns if col != sDepVar]

dfDataFinishedTrainIndep = dfDataFinishedTrain[colIndepVar]
dfDataFinishedTestIndep = dfDataFinishedTest[colIndepVar]

dfDataFinishedTrainDep = dfDataFinishedTrain[['date', 'job_no', sDepVar]]
dfDataFinishedTestDep = dfDataFinishedTest[['date', 'job_no', sDepVar]]