for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import os
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
# sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)

exec(open("Scripts/1_EDA.py").read())

print("#############################################################################################################")
print("########################################## Finished Getting Data ############################################")
print("#############################################################################################################")

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
fraction = (dfDataFinished.groupby('job_no', observed=True).filter(lambda x: len(x) < 6)[
                'job_no'].nunique() / nFinished) * 100
print(f"The fraction of finished jobs with less than 6 observations is {round(fraction, 2)}%.")

# Omit finished jobs with less than 6 observations
dfDataFinished = dfDataFinished.groupby('job_no').filter(lambda x: len(x) >= 6)

# Split the finished jobs into train and test (Assuming 'train' column exists and is binary)
dfDataFinishedTrain = dfDataFinished[dfDataFinished['train'] == 1]
dfDataFinishedTest = dfDataFinished[~dfDataFinished.index.isin(dfDataFinishedTrain.index)]

# Scale all numeric columns
numeric_cols = dfDataFinishedTrain.select_dtypes(include=[np.number]).columns.tolist()

train_data = dfDataFinishedTrain[numeric_cols]
test_data = dfDataFinishedTest[numeric_cols]

# Replace infinite values with NaN
train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
test_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Scale the data
y_scaler = MinMaxScaler()
x_scaler = MinMaxScaler()

# Split into dependent and independent variables
sDepVar = 'total_contribution'
colIndepVarNum = [col for col in train_data.columns if col != sDepVar]
colIndepVar = ['date', 'job_no', 'department'] + colIndepVarNum

train_data_X = train_data[colIndepVarNum]
test_data_X = test_data[colIndepVarNum]

train_data_y = train_data[[sDepVar]]
test_data_y = train_data[[sDepVar]]

# Scale the independent variables
x_scaler = x_scaler.fit(train_data_X)
y_scaler = y_scaler.fit(train_data_y)

train_data_X_scaled = x_scaler.transform(train_data_X)
test_data_X_scaled = x_scaler.transform(test_data_X)

train_data_y_scaled = y_scaler.transform(train_data_y)
test_data_y_scaled = y_scaler.transform(test_data_y)

# Join job_no, date and department to the scaled data (independent variables)
# Get dimensions of train_data_X_scaled and test_data_X_scaled
# Reset index of dfDataFinishedTrain and dfDataFinishedTest
dfDataFinishedTrain = dfDataFinishedTrain.reset_index(drop=True)
dfDataFinishedTest = dfDataFinishedTest.reset_index(drop=True)



train_data_X_scaled = pd.DataFrame(train_data_X_scaled, columns=colIndepVarNum)
train_data_X_scaled = pd.concat([dfDataFinishedTrain[['date', 'job_no', 'department']], train_data_X_scaled], axis=1)
test_data_X_scaled = pd.DataFrame(test_data_X_scaled, columns=colIndepVarNum)
test_data_X_scaled = pd.concat([dfDataFinishedTest[['date', 'job_no', 'department']], test_data_X_scaled], axis=1)

# Join job_no, date and department to the scaled data (dependent variable)
train_data_y_scaled = pd.DataFrame(train_data_y_scaled, columns=[sDepVar])
train_data_y_scaled = pd.concat([dfDataFinishedTrain[['date', 'job_no', 'department']], train_data_y_scaled], axis=1)
test_data_y_scaled = pd.DataFrame(test_data_y_scaled, columns=[sDepVar])
test_data_y_scaled = pd.concat([dfDataFinishedTest[['date', 'job_no', 'department']], test_data_y_scaled], axis=1)

# To reverse the scaling, use the following code:
# train_data_X_scaled = x_scaler.inverse_transform(train_data_X_scaled)
# test_data_X_scaled = x_scaler.inverse_transform(test_data_X_scaled)
# train_data_y_scaled = y_scaler.inverse_transform(train_data_y_scaled)
# test_data_y_scaled = y_scaler.inverse_transform(test_data_y_scaled)

# Merge the dependent and independent variables
train_data_scaled = pd.merge(train_data_y_scaled, train_data_X_scaled, on=['date', 'job_no'])
test_data_scaled = pd.merge(test_data_y_scaled, test_data_X_scaled, on=['date', 'job_no'])

# Save the scaled data
train_data_scaled.to_parquet('./dfDataTrain.parquet')
test_data_scaled.to_parquet('./dfDataTest.parquet')

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

# Only keep numeric columns
train_data_X_scaled = train_data_X_scaled.select_dtypes(include=[np.number])

# Replace infinite values with NaN
train_data_X_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaN with 0
train_data_X_scaled.fillna(0, inplace=True)

# Run PCA
pca = PCA(n_components=20)
pca.fit(train_data_X_scaled.select_dtypes(include=[np.number]))

# Plot the explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()
