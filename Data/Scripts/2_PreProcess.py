import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from plot_config import *
from plot_predicted import *
import joblib

pd.options.mode.chained_assignment = None  # default='warn'

# Load ./dfData.parquet
if os.name == 'posix':
    sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
# If operating system is Windows then
elif os.name == 'nt':
    sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)

import dropbox
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import re
import subprocess

# Read dfData parquet file
dfData = pd.read_parquet("dfData.parquet")

# Replace NA with 0
dfData = dfData.replace(np.nan, 0)

# Split data into wip and finished jobs. This serves as test and train data.
dfDataFinished = dfData[dfData['wip'] == 0]
dfDataWIP = dfData[dfData['wip'] == 1]

### Training Method ###
# Load trainMethod from ./.AUX/trainMethod.txt
with open('./.AUX/trainMethod.txt', 'r') as f:
    trainMethod = f.read()

## Only look at finished jobs in the beginning. Set to dfData when code is ready.
dfData = dfDataFinished.copy()

# Fraction of finished jobs with less than 6 observations
nFinished = len(dfData['job_no'].unique())
fraction = (dfData.groupby('job_no', observed=True).filter(lambda x: len(x) < 6)[
                'job_no'].nunique() / nFinished) * 100
print(f"The fraction of jobs with less than 6 observations is {round(fraction, 2)}%.")

# Omit finished jobs with less than 6 observations
dfData = dfData.groupby('job_no').filter(lambda x: len(x) >= 6)

# Number of unique finished jobs and WIP jobs
nFinished = len(dfData['job_no'].unique())
nWIP = len(dfDataWIP['job_no'].unique())
print(f"The finished jobs account for {round(nFinished / (nFinished + nWIP) * 100, 2)}% of the total number of jobs.")

# Average, Max and Min number of observations per finished job
obs = dfData.groupby('job_no', observed=True).size()
print(f"The average number of observations per finished job is {round(obs.mean(), 2)}.")
print(f"The max number of observations per finished job is {obs.max()}.")
print(f"The min number of observations per finished job is {obs.min()}.")

# Replace infinite values with NaN
dfData.replace([np.inf, -np.inf], np.nan, inplace=True)
dfDataWIP.replace([np.inf, -np.inf], np.nan, inplace=True)

# Scale all numeric columns
numeric_cols = dfData.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = dfData.select_dtypes(exclude=[np.number]).columns.tolist()

# Split into dependent and independent variables
# Read sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

# Remove sDepVar from independent variables
colIndepVarNum = [col for col in dfData[numeric_cols].columns if col != sDepVar]

# Shift all independent variables three periods back, so that the independent variables are lagged.
# This is done to avoid leakage, and to ensure that the model can be used for forecasting.
# dfData[colIndepVarNum] = dfData[colIndepVarNum].shift(1)

# Split the finished jobs into train and test
dfDataTrain = dfData[dfData[trainMethod] == 1]

# Rows in train and test
print(f"The number of rows in train is {len(dfDataTrain)}.")

# Split into numeric and descriptive columns
train_data = dfDataTrain[numeric_cols]
train_data_desc = dfDataTrain[non_numeric_cols]

# Scale the data
y_scaler = MinMaxScaler()
x_scaler = MinMaxScaler()

# x_scaler = PowerTransformer(method='yeo-johnson')
# y_scaler = PowerTransformer(method='yeo-johnson')

# omit 'train', 'train_TS', 'cluster_2', 'cluster_4', 'cluster_6', 'cluster_8', 'cluster_10', 'cluster_12', 'cluster_14'
# from colIndepVarNum
colIndepVarNum = [col for col in colIndepVarNum if
                    col not in ['train', 'train_TS', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5']]
# colIndepVarNum value to file
with open('./.AUX/colIndepVarNum.txt', 'w') as f:
    f.write('\n'.join(colIndepVarNum))

train_data_X = train_data[colIndepVarNum]
train_data_y = train_data[[sDepVar]]

# Scale the independent variables
x_scaler = x_scaler.fit(train_data_X)
y_scaler = y_scaler.fit(train_data_y)

# Change customer zip to string
dfData['customer_zip'] = dfData['customer_zip'].astype(str)
dfData['zip'] = dfData['zip'].astype(str)

dfData_reg = dfData.copy() # unscaled dfData

# For col in colIndepVarNum scale dfData using x_scaler
dfData[colIndepVarNum] = x_scaler.transform(dfData[colIndepVarNum])
dfData[sDepVar] = y_scaler.transform(dfData[[sDepVar]])
dfDataWIP[colIndepVarNum] = x_scaler.transform(dfDataWIP[colIndepVarNum])
dfDataWIP[sDepVar] = y_scaler.transform(dfDataWIP[[sDepVar]])

from sklearn.cluster import KMeans
# Run the process for five different numbers of clusters.
lCluster = [2, 3, 4, 5]
# Assign cluster to each job
for nCluster in lCluster:
    # Create KMeans object
    kmeans = KMeans(n_clusters=nCluster, random_state=607, n_init='auto', max_iter=1000)
    # Fit the model.
    kmeans.fit(dfData[['final_estimate_contribution']].replace(np.nan, 0))
    # Predict the cluster for each observation
    dfData[f'cluster_{nCluster}'] = kmeans.predict(dfData[['final_estimate_contribution']].replace(np.nan, 0))
    dfDataWIP[f'cluster_{nCluster}'] = kmeans.predict(dfDataWIP[['final_estimate_contribution']].replace(np.nan, 0))
    dfData_reg[f'cluster_{nCluster}'] = kmeans.predict(dfData_reg[['final_estimate_contribution']].replace(np.nan, 0))

# Plot number of observations in each cluster in a subplot for each number of clusters
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
for i, nCluster in enumerate(lCluster):
    sns.countplot(x=f'cluster_{nCluster}', data=dfData, ax=ax[i // 2, i % 2])
    ax[i // 2, i % 2].set_xlabel(f'Cluster for {nCluster} cluster solution')
    ax[i // 2, i % 2].set_ylabel('Number of observations')
plt.savefig("./Results/Figures/1_9_cluster.png")
plt.savefig("./Results/Presentation/1_9_cluster.svg")
upload(plt, 'Project-based Internship', 'figures/1_9_cluster.png')

# Add intercept to dfData
dfData['intercept'] = 1
dfDataWIP['intercept'] = 1

# Save dfData to parquet as dfData_scaled
dfData.to_parquet('./dfData_reg_scaled.parquet')
dfDataWIP.to_parquet('./dfData_reg_scaled_wip.parquet')
dfData.to_parquet('./dfData_reg.parquet')

# Save the scales to .AUX/
joblib.dump(x_scaler, "./.AUX/x_scaler.save")
joblib.dump(y_scaler, "./.AUX/y_scaler.save")

plt.close('all')
