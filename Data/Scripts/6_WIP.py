# Import required libraries
import os
import warnings
import numpy as np
import pandas as pd
# import mlforecast
# import neuralforecast
# import keras
import datetime
import joblib
from plot_config import *
from plot_predicted import *
from notify import *
from sklearn.metrics import mean_squared_error
import multiprocessing
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import keras_tuner as kt

warnings.filterwarnings('ignore')

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

# Load data
dfDataWIP = pd.read_parquet("./dfData_reg_scaled_wip.parquet")
dfDataPred = pd.read_parquet("./dfDataPred.parquet")


sJobNo = 'S898024'
sJobNo = 'S283193'
sJobNo = 'S283191'

# Get the data of job_no
dfDataJob = dfDataPred[dfDataPred['job_no'] == sJobNo]








########################################################################################################################


# Import lNumericCols from ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'r') as f:
    lNumericCols = f.read()
lNumericCols = lNumericCols.split('\n')

# Replace infinite values with NaN
dfDataWIP.replace([np.inf, -np.inf], np.nan, inplace=True)

# Replace NaN with 0
dfDataWIP[lNumericCols].fillna(0, inplace=True)

# Import scales
x_scaler = joblib.load("./.AUX/x_scaler.save")
y_scaler = joblib.load("./.AUX/y_scaler.save")

# Import sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

# Import colIndepVarNum from ./.AUX/colIndepVarNum.txt
with open('./.AUX/colIndepVarNum.txt', 'r') as f:
    colIndepVarNum = f.read()
colIndepVarNum = colIndepVarNum.split('\n')

# Import sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

# Load trainMethod from ./.AUX/trainMethod.txt
with open('./.AUX/trainMethod.txt', 'r') as f:
    trainMethod = f.read()

# Import lIndepVar_lag_budget from ./.AUX/lIndepVar_lag_budget.txt
with open('./.AUX/lIndepVar_lag_budget.txt', 'r') as f:
    lIndepVar_lag_budget = f.read()
# Convert string to list
lIndepVar_lag_budget = lIndepVar_lag_budget.split('\n')

# Import dfRMSE from ./Results/Tables/3_4_rmse.csv
dfRMSE = pd.read_csv("./Results/Tables/5_1_rmse.csv", index_col=0)

# Rescale dfDataScaled to dfData
#dfDataWIP[colIndepVarNum] = x_scaler.inverse_transform(dfDataWIP[colIndepVarNum])

# Top 5 models based on smape
lModels = list(dfRMSE.sort_values(by='sMAPE').head(5).index)

## Load models from ./.MODS
# For each model in ./.MODS
for model in os.listdir("./.MODS"):
    # Load model
    model = joblib.load(f"./.MODS/{model}")
    # Predict
    dfDataWIP[f"{model}"] = model.predict(dfDataWIP[colIndep])
    # Rescale
    dfDataWIP[f"{model}"] = y_scaler.inverse_transform(dfDataWIP[f"{model}"].values.reshape(-1, 1))

# Load lstm from ./.MODS/LSTM_tune.tf
lstm = load_model("./.MODS/LSTM_tune.tf")

# Use saved model to predict out of sample data. The model is trained on finished jobs, and the prediction is made on
# the WIP jobs.
# Predict
dfDataWIP['LSTM'] = pd.DataFrame(
    # Ignore index and get the last value of the prediction
    lstm.predict(dfDataWIP[lNumericCols])[:, -1, 0].reshape(-1, 1),
    index=dfDataWIP.index
)

# Rescale
dfDataWIP["LSTM"] = y_scaler.inverse_transform(dfDataWIP["LSTM"].values.reshape(-1, 1))

job_no = 'S283202'
# Get the data of job_no
dfDataJob = dfDataWIP[dfDataWIP['job_no'] == job_no]
# Plot the cumsum of actual and predicted contribution of sJobNo
fig, ax = plt.subplots(figsize=(20, 10))
for col in ['contribution_cumsum',
               'production_estimate_contribution',
               'final_estimate_contribution',
               'LSTM',
               'risk']:
    if col == 'production_estimate_contribution':
        ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashed')
    elif col == 'final_estimate_contribution':
        ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashed')
    elif col == 'risk':
        ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashdot')
    else:
        ax.plot(dfDataJob['date'], dfDataJob[col], label=col)
ax.set_xlabel('Date')
ax.set_ylabel('Contribution')
ax.set_title(f'Actual vs. Predicted Total Contribution of {job_no}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.show()


### Calculate mean of all models ###
# Create list of models to include
for model in os.listdir("./.MODS"):
    if model.endswith(".pickle"):
        lModels = [model for model in os.listdir("./.MODS") if model.endswith(".pickle")]

# Add LSTM to lModels
lModels.append("LSTM")

str = "I'm ready"
# Print read
print(str[4:8])

########################################################################################################################
# if ./Results/Figures/Jobs does not exist, create it

if not os.path.exists('./Results/Figures/WIP'):
    os.makedirs('./Results/Figures/WIP')

## For each job_no plot the actual and predicted sDepVar
for job_no in dfDataWIP['job_no'].unique():
    # Get the data of job_no
    dfDataJob = dfDataWIP[dfDataWIP['job_no'] == job_no]

    dfDataJob['LSTM'] = pd.DataFrame(
        # Ignore index and get the last value of the prediction
        lstm.predict(dfDataJob[lNumericCols])[:, -1, 0].reshape(-1, 1),
        index=dfDataJob.index
    )
    # Rescale
    dfDataJob["LSTM"] = y_scaler.inverse_transform(dfDataJob["LSTM"].values.reshape(-1, 1))

    # Plot the cumsum of actual and predicted contribution of sJobNo
    fig, ax = plt.subplots(figsize=(20, 10))
    for col in ['contribution_cumsum',
                'production_estimate_contribution',
                'final_estimate_contribution',
                'LSTM',
                'risk']:
        if col == 'production_estimate_contribution':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashed')
        elif col == 'final_estimate_contribution':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashed')
        elif col == 'risk':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashdot')
        else:
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution')
    ax.set_title(f'Actual vs. Predicted Total Contribution of {job_no}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.5)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/WIP/{job_no}.png")
    plt.close('all')

########################################################################################################################
