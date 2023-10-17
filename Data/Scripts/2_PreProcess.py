import os
import runpy
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
import joblib

pd.options.mode.chained_assignment = None  # default='warn'

# Load ./dfData.parquet
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
# sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)

# Read dfData parquet file
dfData = pd.read_parquet("dfData.parquet")

# Split data into wip and finished jobs. This serves as test and train data.
dfDataFinished = dfData[dfData['wip'] == 0]
dfDataWIP = dfData[dfData['wip'] == 1]

# Fraction of finished jobs with less than 6 observations
nFinished = len(dfDataFinished['job_no'].unique())
fraction = (dfDataFinished.groupby('job_no', observed=True).filter(lambda x: len(x) < 6)[
                'job_no'].nunique() / nFinished) * 100
print(f"The fraction of finished jobs with less than 6 observations is {round(fraction, 2)}%.")

# Omit finished jobs with less than 6 observations
dfDataFinished = dfDataFinished.groupby('job_no').filter(lambda x: len(x) >= 6)


# Number of unique finished jobs and WIP jobs
nFinished = len(dfDataFinished['job_no'].unique())
nWIP = len(dfDataWIP['job_no'].unique())
print(f"The finished jobs account for {round(nFinished / (nFinished + nWIP) * 100, 2)}% of the total number of jobs.")

# Average, Max and Min number of observations per finished job
obs = dfDataFinished.groupby('job_no', observed=True).size()
print(f"The average number of observations per finished job is {round(obs.mean(), 2)}.")
print(f"The max number of observations per finished job is {obs.max()}.")
print(f"The min number of observations per finished job is {obs.min()}.")

# Split the finished jobs into train and test (Assuming 'train' column exists and is binary)
dfDataFinishedTrain = dfDataFinished[dfDataFinished['train'] == 1]
dfDataFinishedTest = dfDataFinished[~dfDataFinished.index.isin(dfDataFinishedTrain.index)]

# Rows in train and test
print(f"The number of rows in train is {len(dfDataFinishedTrain)}.")
print(f"The number of rows in test is {len(dfDataFinishedTest)}.")

# Omit train column
dfDataFinishedTrain.drop('train', axis=1, inplace=True)
dfDataFinishedTest.drop('train', axis=1, inplace=True)

# Scale all numeric columns
numeric_cols = dfDataFinishedTrain.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = dfDataFinishedTrain.select_dtypes(exclude=[np.number]).columns.tolist()

# Split into numeric and descriptive columns
train_data = dfDataFinishedTrain[numeric_cols]
train_data_desc = dfDataFinishedTrain[non_numeric_cols]
test_data = dfDataFinishedTest[numeric_cols]
test_data_desc = dfDataFinishedTest[non_numeric_cols]

# Replace infinite values with NaN
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
dfDataWIP.replace([np.inf, -np.inf], np.nan, inplace=True)

# Scale the data
y_scaler = MinMaxScaler()
x_scaler = MinMaxScaler()

# Split into dependent and independent variables
sDepVar = 'contribution'
colIndepVarNum = [col for col in train_data.columns if col != sDepVar]
colIndepVar = ['date', 'job_no', 'department'] + colIndepVarNum

train_data_X = train_data[colIndepVarNum]
test_data_X = test_data[colIndepVarNum]

train_data_y = train_data[[sDepVar]]
test_data_y = train_data[[sDepVar]]

# Scale the independent variables
x_scaler = x_scaler.fit(train_data_X)
y_scaler = y_scaler.fit(train_data_y)

# Transform dfDataWIP using the same scaler
dfDataWIP_X = dfDataWIP[colIndepVarNum]
dfDataWIP_X_scaled = pd.DataFrame(x_scaler.transform(dfDataWIP_X), columns=colIndepVarNum)
dfDataWIP_desc = dfDataWIP[['date', 'job_no', 'department']].reset_index(drop=True)
dfDataWIP_X_scaled = pd.concat([dfDataWIP_desc, dfDataWIP_X_scaled], axis=1)

dfDataWIP_y = dfDataWIP[[sDepVar]]
dfDataWIP_y_scaled = pd.DataFrame(y_scaler.transform(dfDataWIP_y), columns=[sDepVar])
dfDataWIP_y_scaled = pd.concat([dfDataWIP_desc, dfDataWIP_y_scaled], axis=1)

# Join X and y on date and job_no
dfDataWIP_scaled = pd.merge(dfDataWIP_X_scaled,
                            dfDataWIP_y_scaled,
                            on=['date', 'job_no', 'department'])
# Fill NaN with 0
dfDataWIP_scaled.fillna(0, inplace=True)

# Save dfDataWIP_scaled to parquet
dfDataWIP_scaled.to_parquet('./dfDataWIP.parquet')

# Join job_no, date and department to the scaled data (independent variables)
# Get dimensions of train_data_X_scaled and test_data_X_scaled
# Reset index of dfDataFinishedTrain and dfDataFinishedTest
dfDataFinishedTrain = dfDataFinishedTrain.reset_index(drop=True)
dfDataFinishedTest = dfDataFinishedTest.reset_index(drop=True)

train_data_X_scaled = pd.DataFrame(x_scaler.transform(train_data_X), columns=colIndepVarNum)
train_data_X_scaled = pd.concat([dfDataFinishedTrain[['date', 'job_no', 'department']], train_data_X_scaled], axis=1)
test_data_X_scaled = pd.DataFrame(x_scaler.transform(test_data_X), columns=colIndepVarNum)
test_data_X_scaled = pd.concat([dfDataFinishedTest[['date', 'job_no', 'department']], test_data_X_scaled], axis=1)

# Join job_no, date and department to the scaled data (dependent variable)
train_data_y_scaled = pd.DataFrame(y_scaler.transform(train_data_y), columns=[sDepVar])
train_data_y_scaled = pd.concat([dfDataFinishedTrain[['date', 'job_no', 'department']], train_data_y_scaled], axis=1)
test_data_y_scaled = pd.DataFrame(y_scaler.transform(test_data_y), columns=[sDepVar])
test_data_y_scaled = pd.concat([dfDataFinishedTest[['date', 'job_no', 'department']], test_data_y_scaled], axis=1)

# To reverse the scaling, use the following code:
# train_data_X_scaled = x_scaler.inverse_transform(train_data_X_scaled)
# test_data_X_scaled = x_scaler.inverse_transform(test_data_X_scaled)
# train_data_y_scaled = y_scaler.inverse_transform(train_data_y_scaled)
# test_data_y_scaled = y_scaler.inverse_transform(test_data_y_scaled)

# Merge the dependent and independent variables
train_data_scaled = pd.merge(train_data_y_scaled, train_data_X_scaled, on=['date', 'job_no'])
test_data_scaled = pd.merge(test_data_y_scaled, test_data_X_scaled, on=['date', 'job_no'])


test_data_scaled.fillna(0, inplace=True)
train_data_scaled.fillna(0, inplace=True)

# Merge train_data_scaled, test_data_scaled and dfDataWIP_scaled
dfDataScaled = pd.concat([train_data_scaled, test_data_scaled, dfDataWIP_scaled], axis=0)

# Save the scaled data
train_data_scaled.to_parquet('./dfDataTrain.parquet')
test_data_scaled.to_parquet('./dfDataTest.parquet')
dfDataScaled.to_parquet('./dfDataScaled.parquet')

# sDepVar value to file
with open('./.AUX/sDepVar.txt', 'w') as f:
    f.write(sDepVar)

# Save the scales to .AUX/
joblib.dump(x_scaler, "./.AUX/x_scaler.save")
joblib.dump(y_scaler, "./.AUX/y_scaler.save")


### Principal Component Analysis ###
# Run PCA on train_data_X_scaled and plot the explained variance ratio.
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_config import *
# Load train_data_X_scaled
train_data_X_scaled = pd.read_parquet('./dfDataTrain.parquet')
train_data_y_scaled = train_data_X_scaled[['total_contribution']]
# Drop total_contribution from train_data_X_scaled
train_data_X_scaled.drop('total_contribution', axis=1, inplace=True)

# Only keep numeric columns
train_data_X_scaled = train_data_X_scaled.select_dtypes(include=[np.number])

# Replace NaN with 0
train_data_X_scaled.fillna(0, inplace=True)

# Run PCA
pca = PCA(n_components=0.99)
pca.fit(train_data_X_scaled.select_dtypes(include=[np.number]))

# Plot the explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()

# Plot first three principal components in 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from plot_config import *

# Plot first three principal components in 3D
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca.components_[0], pca.components_[1], pca.components_[2], s=20, c=vColors[0], marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.tight_layout()
plt.show()

