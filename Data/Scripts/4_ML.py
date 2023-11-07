# Import required libraries
import warnings
import os
import numpy as np
import pandas as pd
import datetime
import joblib
from plot_config import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

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

def upload(ax, project, path):
    bs = BytesIO()
    format = path.split('.')[-1]

    # Check if the file is a .tex file and handle it differently
    if format == 'tex':
        # Assuming the 'ax' parameter contains the LaTeX content
        content = ax
        format = 'tex'
    else:
        ax.savefig(bs, bbox_inches='tight', format=format)

    # token = os.DROPBOX
    token = subprocess.run("curl https://api.dropbox.com/oauth2/token -d grant_type=refresh_token -d refresh_token=eztXuoP098wAAAAAAAAAAV4Ef4mnx_QpRaiqNX-9ijTuBKnX9LATsIZDPxLQu9Nh -u a415dzggdnkro3n:00ocfqin8hlcorr", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.split('{"access_token": "')[1].split('", "token_type":')[0]
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    if format == 'tex':
        # Handle .tex files by directly uploading their content
        dbx.files_upload(content.encode(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)
    else:
        dbx.files_upload(bs.getvalue(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)




# Load data
dfDataScaled = pd.read_parquet("./dfData_reg_scaled.parquet")
dfData = pd.read_parquet("./dfData_reg.parquet")

# Load dfDataPred from ./dfDataPred.parquet
dfDataPred = pd.read_parquet("./dfDataPred.parquet")


# Define sMAPE
def smape(actual, predicted):
    return 100 / len(actual) * np.sum(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


# Load trainMethod from ./.AUX/trainMethod.txt
with open('./.AUX/trainMethod.txt', 'r') as f:
    trainMethod = f.read()

# Import lNumericCols from ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'r') as f:
    lNumericCols = f.read()
lNumericCols = lNumericCols.split('\n')

# Replace infinite values with NaN
dfDataScaled.replace([np.inf, -np.inf], np.nan, inplace=True)
dfData.replace([np.inf, -np.inf], np.nan, inplace=True)

# If trainMethod == 'train', then oTrain = 'train_TS' else oTrain = 'train'
if trainMethod == 'train':
    oTrain = 'train_TS'
else:
    oTrain = 'train'


# Replace NaN with 0
dfDataScaled.select_dtypes(include=['float64', 'int64']).fillna(0, inplace=True)
dfData.select_dtypes(include=['float64', 'int64']).fillna(0, inplace=True)

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

# Import lDST from ./.AUX/lDST.txt
with open('./.AUX/lDST.txt', 'r') as f:
    lDST = f.read()
lDST = lDST.split('\n')

# Rescale dfDataScaled to dfData
dfDataRescaled = dfDataScaled.copy()
dfDataRescaled[colIndepVarNum] = x_scaler.inverse_transform(dfDataScaled[colIndepVarNum].values)
dfDataRescaled[sDepVar] = y_scaler.inverse_transform(dfDataScaled[sDepVar].values.reshape(-1, 1))

train_index = dfData[dfData[trainMethod] == 1].index
dfDataScaledTrain = dfDataScaled.loc[train_index]
dfDataScaledTest = dfDataScaled.drop(train_index)

# Import sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

# Import lIndepVar_lag_budget from ./.AUX/lIndepVar_lag_budget.txt
with open('./.AUX/lIndepVar_lag_budget.txt', 'r') as f:
    lIndepVar_lag_budget = f.read()
# Convert string to list
lIndepVar_lag_budget = lIndepVar_lag_budget.split('\n')

# Import dfRMSE from ./Results/Tables/3_4_rmse.csv
dfRMSE = pd.read_csv("./Results/Tables/3_4_rmse.csv", index_col=0)

### Elastic Net Regression ###
# Define Elastic Net model

# Define hyperparameter grid
param_grid_en = {
    'alpha': np.arange(0.1, 10, 0.01),
    'l1_ratio': np.arange(0.001, 1.00, 0.001),
    'tol': [0.0001, 0.001, 0.01]
}
# Define randomized search
elastic_net = ElasticNet(tol=1e-4, random_state=0)
elastic_net_cv_sparse = RandomizedSearchCV(elastic_net, param_grid_en, n_iter=1000, scoring=None, cv=3, verbose=0,
                                           refit=True, n_jobs=-1)

# get unique entries in lIndepVar_lag_budget + lDST
lIndepVar = list(set(lIndepVar_lag_budget)) + list(set(lDST))
# Drop duplicates from lIndepVar
lIndepVar = list(dict.fromkeys(lIndepVar))
# If lIndepVar contains like 'cluster_' then remove it
lIndepVar = [col for col in lIndepVar if not col.startswith('cluster_')]

# Save lIndepVar to ./.AUX/lIndepVar.txt
with open('./.AUX/lIndepVar.txt', 'w') as f:
    f.write('\n'.join(lIndepVar))

lIndepVar = lIndepVar + ['intercept']
# Sparse model with OLS variables
start_time_en_sparse = datetime.datetime.now()
elastic_net_cv_sparse.fit(dfDataScaledTrain[lIndepVar].replace(np.nan, 0), dfDataScaledTrain[sDepVar])
# Save model to .MODS/ as pickle
joblib.dump(elastic_net_cv_sparse, './.MODS/elastic_net_cv_sparse.pickle')
dfData['predicted_en_sparse'] = elastic_net_cv_sparse.predict(dfDataScaled[lIndepVar].replace(np.nan, 0))
dfData['predicted_en_sparse'] = y_scaler.inverse_transform(dfData['predicted_en_sparse'].shift(-1).values.reshape(-1, 1))
end_time_en_sparse = datetime.datetime.now()
print(f'ElasticNet finished in {end_time_en_sparse - start_time_en_sparse}.')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_en_sparse'].transform('sum'),
        label='Predicted (Elastic Net)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_0_en.png")
plt.savefig("./Results/Presentation/4_0_en.svg")
upload(plt, 'Project-based Internship', 'figures/4_0_en.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_en_sparse'].transform('sum'),
        label='Predicted (Elastic Net)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_0_1_en.png")
plt.savefig("./Results/Presentation/4_0_1_en.svg")
upload(plt, 'Project-based Internship', 'figures/4_0_1_en.png')
plt.close('all')

# Calculate RMSE of EN
rmse_en_sparse = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_en_sparse'].replace(np.nan, 0)))
# Calculate sMAPE
smape_en_sparse = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                        dfData[dfData[trainMethod] == 0]['predicted_en_sparse'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Elastic Net', 'RMSE'] = rmse_en_sparse
dfRMSE.loc['Elastic Net', 'sMAPE'] = smape_en_sparse

# Add to dfDataPred
dfDataPred['predicted_en_sparse'] = dfData['predicted_en_sparse']

### Random Forest Regression ###
# Define Random Forest model
rf = RandomForestRegressor(n_jobs=-1, random_state=0, verbose=False)

## Define hyperparameter grid ##
rf_grid = {
    "n_estimators": np.arange(10, 300, 20),  # Number of trees
    "max_depth": [None, 1, 3, 5, 10, 15, 20, 25],  # Depth of each tree
    "min_samples_split": np.arange(2, 60, 2),  # Minimum samples required to split an internal node
    "min_samples_leaf": np.arange(1, 60, 2),  # Minimum samples required to be at a leaf node
    "max_features": [1 / 3, 0.5, 1, "sqrt", "log2", None],  # Number of features to consider for split
    "max_samples": [100, 150, 250, 500, 1000, 1500, None]  # Number of samples to train each tree
}

# Define randomized search
rf_cv = RandomizedSearchCV(rf, rf_grid, n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error", cv=3, verbose=False,
                           refit=True)
# Fit to the training data
start_time_rf = datetime.datetime.now()
rf_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
          dfDataScaledTrain[sDepVar])
# Save model to .MODS/ as pickle
joblib.dump(rf_cv, './.MODS/rf_cv.pickle')
# Predict and rescale using RF
dfData['predicted_rf_full'] = rf_cv.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_rf_full'] = y_scaler.inverse_transform(dfData['predicted_rf_full'].shift(-1).values.reshape(-1, 1))
end_time_rf = datetime.datetime.now()
print(f'RF Full fit finished in {end_time_rf - start_time_rf}.')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_rf_full'].transform('sum'),
        label='Predicted (Full Random Forest)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_1_rf_full.png")
plt.savefig("./Results/Presentation/4_1_rf_full.svg")
upload(plt, 'Project-based Internship', 'figures/4_1_rf_full.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_rf_full'].transform('sum'),
        label='Predicted (Full Random Forest)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_1_1_rf_full.png")
plt.savefig("./Results/Presentation/4_1_1_rf_full.svg")
upload(plt, 'Project-based Internship', 'figures/4_1_1_rf_full.png')

# Calculate RMSE of RF
rmse_rf_full = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_rf_full'].replace(np.nan, 0)))
# Calculate sMAPE
smape_rf_full = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfData[dfData[trainMethod] == 0]['predicted_rf_full'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Random Forest (Full)', 'RMSE'] = rmse_rf_full
dfRMSE.loc['Random Forest (Full)', 'sMAPE'] = smape_rf_full

# Add to dfDataPred
dfDataPred['predicted_rf_full'] = dfData['predicted_rf_full']

# Random Forest using only lIndepVar
# Define randomized search
rf_cv = RandomizedSearchCV(rf, rf_grid, n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error", cv=3, verbose=False,
                           refit=True)
# Fit to the training data
start_time_rf = datetime.datetime.now()
rf_cv.fit(dfDataScaledTrain[lIndepVar][dfDataScaledTrain[lIndepVar].columns.difference([sDepVar])],
          dfDataScaledTrain[sDepVar])
# Save model to .MODS/ as pickle
joblib.dump(rf_cv, './.MODS/rf_cv_sparse.pickle')
# Predict and rescale using RF
dfData['predicted_rf_sparse'] = rf_cv.predict(
    dfDataScaled[lIndepVar][dfDataScaled[lIndepVar].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_rf_sparse'] = y_scaler.inverse_transform(dfData['predicted_rf_sparse'].shift(-1).values.reshape(-1, 1))
end_time_rf = datetime.datetime.now()
print(f'RF Sparse fit finished in {end_time_rf - start_time_rf}.')

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_rf_sparse'].transform('sum'),
        label='Predicted (Sparse Random Forest)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_1_rf_sparse.png")
plt.savefig("./Results/Presentation/4_1_rf_sparse.svg")
upload(plt, 'Project-based Internship', 'figures/4_1_rf_sparse.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_rf_sparse'].transform('sum'),
        label='Predicted (Sparse Random Forest)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_1_1_rf_sparse.png")
plt.savefig("./Results/Presentation/4_1_1_rf_sparse.svg")
upload(plt, 'Project-based Internship', 'figures/4_1_1_rf_sparse.png')

# Calculate RMSE of RF
rmse_rf_sparse = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_rf_sparse'].replace(np.nan, 0)))
# Calculate sMAPE
smape_rf_sparse = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                        dfData[dfData[trainMethod] == 0]['predicted_rf_sparse'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Random Forest (Sparse)', 'RMSE'] = rmse_rf_sparse
dfRMSE.loc['Random Forest (Sparse)', 'sMAPE'] = smape_rf_sparse

# Add to dfDataPred
dfDataPred['predicted_rf_sparse'] = dfData['predicted_rf_sparse']

### Boosted Regression Trees ###

# Define Boosted Regression Trees model
from sklearn.ensemble import GradientBoostingRegressor

# Set random grid for Gradient Boosting
gb_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    'max_depth': [None, 1, 3, 5, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12, 14],
    'max_features': [1 / 3, 0.5, 1, "sqrt", "log2", None],
    'n_estimators': [100, 150, 250, 500, 1000, 1500, 2000]
}

# Define randomized search
gb_cv = RandomizedSearchCV(GradientBoostingRegressor(random_state=0), gb_grid, n_iter=100, scoring=None, cv=3,
                           verbose=0, refit=True, n_jobs=-1)

# Fit to the training data
start_time_gb = datetime.datetime.now()
gb_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
          dfDataScaledTrain[sDepVar])
# Save model to .MODS/ as pickle
joblib.dump(gb_cv, './.MODS/gb_cv.pickle')
# Predict and rescale using GB
dfData['predicted_gb'] = gb_cv.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_gb'] = y_scaler.inverse_transform(dfData['predicted_gb'].shift(-1).values.reshape(-1, 1))
end_time_gb = datetime.datetime.now()
print(f'GB fit finished in {end_time_gb - start_time_gb}.')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_gb'].transform('sum'),
        label='Predicted (Gradient Boosting)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_2_gb.png")
plt.savefig("./Results/Presentation/4_2_gb.svg")
upload(plt, 'Project-based Internship', 'figures/4_2_gb.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_gb'].transform('sum'),
        label='Predicted (Gradient Boosting)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_2_1_gb.png")
plt.savefig("./Results/Presentation/4_2_1_gb.svg")
upload(plt, 'Project-based Internship', 'figures/4_2_1_gb.png')

# Calculate RMSE of GB
rmse_gb = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_gb'].replace(np.nan, 0)))
# Calculate sMAPE
smape_gb = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_gb'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Gradient Boosting', 'RMSE'] = rmse_gb
dfRMSE.loc['Gradient Boosting', 'sMAPE'] = smape_gb

# Add to dfDataPred
dfDataPred['predicted_gb'] = dfData['predicted_gb']

### XGBoost Regression ###
# Define XGBoost model
from xgboost import XGBRegressor

xgb_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.001, 0.01, 0.1],
    'reg_lambda': [0, 0.001, 0.01, 0.1],
}

xgb_cv = RandomizedSearchCV(XGBRegressor(random_state=0), xgb_grid, n_iter=100, scoring=None, cv=3,
                            verbose=0, refit=True, n_jobs=-1)

# Fit to the training data
start_time_xgb = datetime.datetime.now()
xgb_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
           dfDataScaledTrain[sDepVar])
# Save model to .MODS/ as pickle
joblib.dump(xgb_cv, './.MODS/xgb_cv.pickle')
# Predict and rescale using XGB
dfData['predicted_xgb'] = xgb_cv.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_xgb'] = y_scaler.inverse_transform(dfData['predicted_xgb'].shift(-1).values.reshape(-1, 1))
end_time_xgb = datetime.datetime.now()

print(f'XGB fit finished in {end_time_xgb - start_time_xgb}.')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_xgb'].transform('sum'),
        label='Predicted (XGBoost)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_3_xgb.png")
plt.savefig("./Results/Presentation/4_3_xgb.svg")
upload(plt, 'Project-based Internship', 'figures/4_3_xgb.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_xgb'].transform('sum'),
        label='Predicted (XGBoost)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_3_1_xgb.png")
plt.savefig("./Results/Presentation/4_3_1_xgb.svg")
upload(plt, 'Project-based Internship', 'figures/4_3_1_xgb.png')

# Calculate RMSE of XGB
rmse_xgb = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfData[dfData[trainMethod] == 0]['predicted_xgb'].replace(np.nan, 0)))
# Calculate sMAPE
smape_xgb = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfData[dfData[trainMethod] == 0]['predicted_xgb'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['XGBoost', 'RMSE'] = rmse_xgb
dfRMSE.loc['XGBoost', 'sMAPE'] = smape_xgb

# Add to dfDataPred
dfDataPred['predicted_xgb'] = dfData['predicted_xgb']

### Forecast Combination with Boosting
dfDataPred['predicted_boost'] = (dfDataPred['predicted_gb'] + dfDataPred['predicted_xgb']) / 2
dfData['predicted_boost'] = (dfData['predicted_gb'] + dfDataPred['predicted_xgb']) / 2

# Calculate RMSE of GB_FC
rmse_gb_fc = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfDataPred[dfData[trainMethod] == 0]['predicted_boost'].replace(np.nan, 0)))
# Calculate sMAPE
smape_gb_fc = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfDataPred[dfData[trainMethod] == 0]['predicted_boost'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Boosting (FC)', 'RMSE'] = rmse_gb_fc
dfRMSE.loc['Boosting (FC)', 'sMAPE'] = smape_gb_fc

# Save dfDataPred to ./dfDataPred.parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)
print(dfRMSE)

dfRMSE.to_csv("./Results/Tables/3_4_rmse.csv")

plt.close('all')

########################################################################################################################

# For each col in dfDataPred other than date and job_no calculate the running sum.
# Then, for each col in dfDataPred other than date and job_no, calculate the correlation between the running sum and
# the actual sDepVar.

# Create dfDataPredSum
dfDataPredSum = dfDataPred.copy()
# Omit date
dfDataPredSum.sort_values(by=['date'], inplace=True)
dfDataPredSum = dfDataPredSum.drop(columns=['date'])
# Group by job_no and calculate cumulative sum of each column
dfDataPredSum = dfDataPredSum.groupby('job_no').cumsum()
# Add date and job_no
dfDataPredSum['date'] = dfDataPred['date']
dfDataPredSum['job_no'] = dfDataPred['job_no']

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfDataPredSum['date'],
        dfDataPredSum.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfDataPredSum['date'],
        dfDataPredSum.groupby('date')['predicted_en_sparse'].transform('sum'),
        label='Elastic Net')
ax.plot(dfDataPredSum['date'],
        dfDataPredSum.groupby('date')['predicted_rf_full'].transform('sum'),
        label='Full Random Forest')
ax.plot(dfDataPredSum['date'],
        dfDataPredSum.groupby('date')['predicted_rf_sparse'].transform('sum'),
        label='Sparse Random Forest')
ax.plot(dfDataPredSum['date'],
        dfDataPredSum.groupby('date')['predicted_gb'].transform('sum'),
        label='Gradient Boosting')
ax.plot(dfDataPredSum['date'],
        dfDataPredSum.groupby('date')['predicted_boost'].transform('sum'),
        label='XGBoost')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/4_9_sum.png")
plt.savefig("./Results/Presentation/4_9_sum.svg")
upload(plt, 'Project-based Internship', 'figures/4_9_sum.png')

# Save to .parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")

plt.close('all')
