# Import required libraries
import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
import joblib
from plot_config import *
from sklearn.metrics import mean_squared_error

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
dfDataWIP = pd.read_parquet("./dfData_reg_scaled_wip.parquet")

# Import scales
x_scaler = joblib.load("./.AUX/x_scaler.save")
y_scaler = joblib.load("./.AUX/y_scaler.save")


# Define sMAPE
def smape(actual, predicted):
    return 100 / len(actual) * np.sum(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


# Import sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

# Import colIndepVarNum from ./.AUX/colIndepVarNum.txt
with open('./.AUX/colIndepVarNum.txt', 'r') as f:
    colIndepVarNum = f.read()

colIndepVarNum = colIndepVarNum.split('\n')

# Load trainMethod from ./.AUX/trainMethod.txt
with open('./.AUX/trainMethod.txt', 'r') as f:
    trainMethod = f.read()

# Rescale dfDataScaled to dfData
dfDataRescaled = dfDataScaled.copy()
dfDataRescaled[colIndepVarNum] = x_scaler.inverse_transform(dfDataScaled[colIndepVarNum].values)
dfDataRescaled[sDepVar] = y_scaler.inverse_transform(dfDataScaled[sDepVar].values.reshape(-1, 1))
dfDataWIP[sDepVar] = y_scaler.inverse_transform(dfDataWIP[sDepVar].values.reshape(-1, 1))

# Get the 5 most correlated variables (of numeric variables)
lNumericCols = dfDataScaled.select_dtypes(include=[np.number]).columns.tolist()

# Write to ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'w') as lVars:
    lVars.write('\n'.join(lNumericCols))

### Split dfDataScaled into train and test ###
# Get index of train from dfData[trainMethod]
train_index = dfData[dfData[trainMethod] == 1].index

# Split dfDataScaled into train and test
dfDataScaledTrain = dfDataScaled.loc[train_index]
dfDataScaledTest = dfDataScaled.drop(train_index)

### Predict sDepVar using OLS ###
# Predict using data from DST (only external variables)

# Get variables starting with kbyg
lDST = [col for col in dfDataScaled.columns if re.match('kbyg', col)]


# Write lDST to .AUX/
with open('./.AUX/lDST.txt', 'w') as lVars:
    lVars.write('\n'.join(lDST))

model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lDST])
results_dst = model.fit()
# Save model to .MODS/
results_dst.save('./.MODS/results_dst.pickle')
# Save results to LaTeX
ols = results_dst.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_0_dst.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

upload(ols, 'Project-based Internship', 'tables/3_0_dst.tex')

# Predict and rescale sDepVar using OLS
dfData['predicted_dst'] = y_scaler.inverse_transform(results_dst.predict(dfDataScaled[lDST]).values.reshape(-1, 1))

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_dst'].transform('sum'), label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_0_dst.png")
plt.savefig("./Results/Presentation/3_0_dst.svg")
upload(plt, 'Project-based Internship', 'figures/3_0_dst.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_dst'].transform('sum'), label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_0_1_dst.png")
plt.savefig("./Results/Presentation/3_0_1_dst.svg")
upload(plt, 'Project-based Internship', 'figures/3_0_1_dst.png')
plt.close('all')

# Calculate out-of-sample RMSE of DST
rmse_dst = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_dst'].replace(np.nan, 0)
                       ))
# symmetric Mean Absolute Error (sMAPE)
smape_dst = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_dst'].replace(np.nan, 0))

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_dst'] = y_scaler.inverse_transform(results_dst.predict(dfDataWIP[lDST]).values.reshape(-1, 1))


### OLS with s-curve differences. ###

lSCurve = ['revenue_scurve_diff','costs_scurve_diff','contribution_scurve_diff']

model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lSCurve])
results_scurve = model.fit()
# Save model to .MODS/
results_scurve.save('./.MODS/results_scurve.pickle')
# Save results to LaTeX
ols = results_scurve.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_9_scurve.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

upload(ols, 'Project-based Internship', 'tables/3_9_scurve.tex')

# Predict and rescale sDepVar using OLS
dfData['predicted_scurve'] = y_scaler.inverse_transform(results_scurve.predict(dfDataScaled[lSCurve]).values.reshape(-1, 1))

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_scurve'].transform('sum'), label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_9_scurve.png")
plt.savefig("./Results/Presentation/3_9_scurve.svg")
upload(plt, 'Project-based Internship', 'figures/3_9_scurve.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_scurve'].transform('sum'), label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_9_1_scurve.png")
plt.savefig("./Results/Presentation/3_9_1_scurve.svg")
upload(plt, 'Project-based Internship', 'figures/3_9_1_scurve.png')
plt.close('all')

# Calculate out-of-sample RMSE of DST
rmse_scurve = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_scurve'].replace(np.nan, 0)
                       ))
# symmetric Mean Absolute Error (sMAPE)
smape_scurve = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_scurve'].replace(np.nan, 0))

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_scurve'] = y_scaler.inverse_transform(results_scurve.predict(dfDataWIP[lSCurve]).values.reshape(-1, 1))


### Using correlation to select variables ###
corr = dfData[dfData[trainMethod] == 1][lNumericCols].corr()
corr = corr.sort_values(by=sDepVar, ascending=False)
corr = corr[sDepVar]
# Filter out variables with "contribution" or "revenue" in the name
corr = corr[~corr.index.str.contains('contribution')]
corr = corr[0:10]
# Save the 5 most correlated variables in a list
lIndepVar = corr.index.tolist()

# Plot correlation between sDepVar and lIndepVar
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(dfData[dfData[trainMethod] == 1][[sDepVar] + lIndepVar].corr(), annot=True, vmin=-1, vmax=1, fmt='.2f',
            cmap=LinearSegmentedColormap.from_list('custom_cmap', [
                (0, vColors[1]),
                (0.5, '#FFFFFF'),
                (1, vColors[0]
                 )])
            )
plt.title(f'Correlation between {sDepVar} and selected variables')
plt.savefig("./Results/Figures/3_0_2_corr.png")
plt.savefig("./Results/Presentation/3_0_2_corr.svg")
upload(plt, 'Project-based Internship', 'figures/3_0_2_corr.png')

# Run OLS
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar], missing='drop')
results_ols = model.fit()
# Save model to .MODS/
results_ols.save('./.MODS/results_ols.pickle')
# Save results to LaTeX
ols = results_ols.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

upload(ols, 'Project-based Internship', 'tables/3_1_ols.tex')

# Predict and rescale sDepVar using OLS
dfData['predicted_ols'] = results_ols.predict(dfDataScaled[lIndepVar])

dfData['predicted_ols'] = y_scaler.inverse_transform(dfData['predicted_ols'].values.reshape(-1, 1))

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_ols'].transform('sum'), label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_1_ols.png")
plt.savefig("./Results/Presentation/3_1_ols.svg")
upload(plt, 'Project-based Internship', 'figures/3_1_ols.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_ols'].transform('sum'), label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_1_1_ols.png")
plt.savefig("./Results/Presentation/3_1_1_ols.svg")
upload(plt, 'Project-based Internship', 'figures/3_1_1_ols.png')
plt.close('all')

# Calculate out-of-sample RMSE of OLS
rmse_ols = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_ols'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_ols'].replace(np.nan, 0))

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_ols'] = y_scaler.inverse_transform(results_ols.predict(dfDataWIP[lIndepVar]).values.reshape(-1, 1))

### Add lagged variables to lIndepVar ###
# Add lagged variables to lIndepVar
lIndepVar_lag = lIndepVar + ['contribution_lag1', 'revenue_lag1', 'costs_lag1',
                             'contribution_lag2', 'revenue_lag2', 'costs_lag2',
                             'contribution_lag3', 'revenue_lag3', 'costs_lag3']

# Correlation between sDepVar and lIndepVar_lag
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(dfData[dfData[trainMethod] == 1][[sDepVar] + lIndepVar_lag].corr(), annot=True, vmin=-1, vmax=1,
            fmt='.2f',
            cmap=LinearSegmentedColormap.from_list('custom_cmap', [
                (0, vColors[1]),
                (0.5, '#FFFFFF'),
                (1, vColors[0]
                 )])
            )
plt.title(f'Correlation between {sDepVar} and selected variables')
plt.savefig("./Results/Figures/3_2_2_corr_incl_lag.png")
plt.savefig("./Results/Presentation/3_2_2_corr_incl_lag.svg")
upload(plt, 'Project-based Internship', 'figures/3_2_2_corr_incl_lag.png')

# Run OLS with lagged variables
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar_lag], missing='drop')
results_ols_lag = model.fit()
# Save model to .MODS/
results_ols_lag.save('./.MODS/results_ols_lag.pickle')
# Save results to LaTeX
ols = results_ols_lag.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_2_ols_lag.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

upload(ols, 'Project-based Internship', 'tables/3_2_ols_lag.tex')

# Predict and rescale sDepVar using OLS with lagged variables
dfData['predicted_lag'] = results_ols_lag.predict(dfDataScaled[lIndepVar_lag])

dfData['predicted_lag'] = y_scaler.inverse_transform(dfData['predicted_lag'].values.reshape(-1, 1))

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_lag'].transform('sum'),
        label='Predicted (incl. lag)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_3_ols_lag.png")
plt.savefig("./Results/Presentation/3_3_ols_lag.svg")
upload(plt, 'Project-based Internship', 'figures/3_3_ols_lag.png')

#
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lag'].transform('sum'),
        label='Predicted (incl. lag)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_3_1_ols_lag.png")
plt.savefig("./Results/Presentation/3_3_1_ols_lag.svg")
upload(plt, 'Project-based Internship', 'figures/3_3_1_ols_lag.png')
plt.close('all')

# Calculate RMSE of OLS with lagged variables
rmse_ols_lag = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_lag'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                      dfData[dfData[trainMethod] == 0]['predicted_lag'].replace(np.nan, 0))

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_lag'] = y_scaler.inverse_transform(results_ols_lag.predict(dfDataWIP[lIndepVar_lag]).values.reshape(-1, 1))

# Include production_estimate_contribution and sales_estimate_contribution
lIndepVar_lag_budget = lIndepVar_lag + ['production_estimate_contribution', 'sales_estimate_contribution']

# Save lIndepVar_lag_budget to .AUX/
with open('./.AUX/lIndepVar_lag_budget.txt', 'w') as lVars:
    lVars.write('\n'.join(lIndepVar_lag_budget))

# Correlation between sDepVar and lIndepVar_lag_budget
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(dfData[dfData[trainMethod] == 1][[sDepVar] + lIndepVar_lag_budget].corr(), annot=True, vmin=-1, vmax=1,
            fmt='.2f',
            cmap=LinearSegmentedColormap.from_list('custom_cmap', [
                (0, vColors[1]),
                (0.5, '#FFFFFF'),
                (1, vColors[0]
                 )])
            )
plt.title(f'Correlation between {sDepVar} and selected variables')
plt.savefig("./Results/Figures/3_4_2_corr_incl_lag_budget.png")
plt.savefig("./Results/Presentation/3_4_2_corr_incl_lag_budget.svg")
upload(plt, 'Project-based Internship', 'figures/3_4_2_corr_incl_lag_budget.png')

# Run OLS with lagged variables and budget
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar_lag_budget], missing='drop')
results_lag_budget = model.fit()
# Save model to .MODS/
results_lag_budget.save('./.MODS/results_lag_budget.pickle')
# Save results to LaTeX
ols = results_lag_budget.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_3_ols_lag_budget.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

upload(ols, 'Project-based Internship', 'tables/3_3_ols_lag_budget.tex')

# Predict and rescale sDepVar using OLS with lagged variables and budget
dfData['predicted_lag_budget'] = results_lag_budget.predict(dfDataScaled[lIndepVar_lag_budget])

dfData['predicted_lag_budget'] = y_scaler.inverse_transform(dfData['predicted_lag_budget'].values.reshape(-1, 1))

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_lag_budget'].transform('sum'),
        label='Predicted (incl. lag and budget)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_4_ols_lag_budget.png")
plt.savefig("./Results/Presentation/3_4_ols_lag_budget.svg")
upload(plt, 'Project-based Internship', 'figures/3_4_ols_lag_budget.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lag_budget'].transform('sum'),
        label='Predicted (incl. lag and budget)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_4_1_ols_lag_budget.png")
plt.savefig("./Results/Presentation/3_4_1_ols_lag_budget.svg")
upload(plt, 'Project-based Internship', 'figures/3_4_1_ols_lag_budget.png')
plt.close('all')

# Calculate RMSE of OLS with lagged variables and budget
rmse_ols_lag_budget = np.sqrt(mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                                                 dfData[dfData[trainMethod] == 0]['predicted_lag_budget'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag_budget = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                             dfData[dfData[trainMethod] == 0]['predicted_lag_budget'].replace(np.nan, 0))

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_lag_budget'] = results_lag_budget.predict(dfDataWIP[lIndepVar_lag_budget])

dfDataWIP['predicted_lag_budget'] = y_scaler.inverse_transform(
    dfDataWIP['predicted_lag_budget'].values.reshape(-1, 1))

### Forecast Combination ###
# Produce a combined forecast of ols_lag_budget and pls
dfData['predicted_fc'] = dfData['predicted_dst']*0.1 + dfData['predicted_lag_budget']*0.45 + dfData['predicted_scurve']*0.45

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_fc'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_5_fc.png")
plt.savefig("./Results/Presentation/3_5_fc.svg")
upload(plt, 'Project-based Internship', 'figures/3_5_fc.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_fc'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)

plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_5_1_fc.png")
plt.savefig("./Results/Presentation/3_5_1_fc.svg")
upload(plt, 'Project-based Internship', 'figures/3_5_1_fc.png')
plt.close('all')

# Calculate RMSE of Forecast Combination
rmse_fc = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_fc'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_fc'])

# Compare RMSE and sMAPE of the different models in a table
dfRMSE = pd.DataFrame({'RMSE': [rmse_dst, rmse_scurve, rmse_ols, rmse_ols_lag, rmse_ols_lag_budget, rmse_fc],
                       'sMAPE': [smape_dst, smape_scurve, smape_ols, smape_ols_lag, smape_ols_lag_budget, smape_fc]},
                      index=['DST', 'S-curve', 'OLS', 'OLS with lagged variables', 'OLS with lagged variables and budget', 'FC'])

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

### Use clustering to find similar jobs and predict sDepVar for each cluster ###
lCluster = [2, 3, 4, 5]
# For each cluster in cluster_{lCluster} do

for iCluster in lCluster:
    # Get the cluster labels to list using value_counts()
    lClusterLabels = dfData['cluster_' + str(iCluster)].value_counts().index.tolist()
    print(lClusterLabels)

    for iClusterLabel in lClusterLabels:
        # Filter the data based on cluster and label
        data_subset = dfDataScaledTrain[(dfDataScaledTrain['cluster_' + str(iCluster)] == iClusterLabel)]

        # Check if the subset of data has enough observations for OLS
        if data_subset.shape[0] > 1:
            # Run OLS
            model_cluster = sm.OLS(data_subset[sDepVar], data_subset[lIndepVar_lag_budget])
            results_cluster = model_cluster.fit()
            # Save model to .MODS/
            results_cluster.save('./.MODS/results_cluster_' + str(iCluster) + '_' + str(iClusterLabel) + '.pickle')
            # Predict and rescale sDepVar using OLS with lagged variables and budget and add to cluster_{iCluster}
            dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
                iCluster)] = results_cluster.predict(
                dfDataScaled[dfDataScaled['cluster_' + str(iCluster)] == iClusterLabel][lIndepVar_lag_budget])
            dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
                iCluster)] = y_scaler.inverse_transform(
                dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
                    iCluster)].values.reshape(-1, 1))
        else:
            print("Not enough data points for OLS in cluster", iCluster, "and label", iClusterLabel)
            dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
                iCluster)] = np.nan



# Plot the sum of all predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
for iCluster in lCluster:
    ax.plot(dfData[dfData[trainMethod] == 0]['date'],
            dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_' + str(iCluster)].transform('sum'),
            label='Predicted (Cluster ' + str(iCluster) + ')')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_6_cluster.png")
plt.savefig("./Results/Presentation/3_6_cluster.svg")
upload(plt, 'Project-based Internship', 'figures/3_6_cluster.png')
plt.close('all')

# Use Forecast Combination to combine the predictions of each cluster
# For each cluster in cluster_{lCluster} do
dfData['predicted_cluster_fc'] = (dfData['predicted_cluster_' + str(lCluster[0])]
                                  + dfData['predicted_cluster_' + str(lCluster[1])]
                                  + dfData['predicted_cluster_' + str(lCluster[2])]
                                  + dfData['predicted_cluster_' + str(lCluster[3])]) / 4

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_fc'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_7_fc_cluster.png")
plt.savefig("./Results/Presentation/3_7_fc_cluster.svg")
upload(plt, 'Project-based Internship', 'figures/3_7_fc_cluster.png')

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'], dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'], dfData.groupby('date')['predicted_cluster_fc'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_7_1_fc_cluster.png")
plt.savefig("./Results/Presentation/3_7_1_fc_cluster.svg")
upload(plt, 'Project-based Internship', 'figures/3_7_1_fc_cluster.png')
plt.close('all')

# Calculate RMSE of Forecast Combination
rmse_fc_cluster = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_cluster_fc'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc_cluster = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                         dfData[dfData[trainMethod] == 0]['predicted_cluster_fc'].replace(np.nan, 0))

# Add RMSE and sMAPE of Forecast Combination to dfRMSE
dfRMSE.loc['FC_cluster'] = [rmse_fc_cluster, smape_fc_cluster]

### Combine Cluster Forecast Combination and DST ###

dfData['predicted_fc_cluster_dst'] = 0.8*dfData['predicted_cluster_fc'] + dfData['predicted_dst']*0.2

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_fc_cluster_dst'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_8_fc_cluster_dst.png")
plt.savefig("./Results/Presentation/3_8_fc_cluster_dst.svg")
upload(plt, 'Project-based Internship', 'figures/3_8_fc_cluster_dst.png')

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'], dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'], dfData.groupby('date')['predicted_fc_cluster_dst'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_8_1_fc_cluster_dst.png")
plt.savefig("./Results/Presentation/3_8_1_fc_cluster_dst.svg")
upload(plt, 'Project-based Internship', 'figures/3_8_1_fc_cluster_dst.png')
plt.close('all')

# Calculate RMSE of Forecast Combination
rmse_fc_cluster_dst = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_fc_cluster_dst'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc_cluster_dst = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                             dfData[dfData[trainMethod] == 0]['predicted_fc_cluster_dst'].replace(np.nan, 0))

# Add RMSE and sMAPE of Forecast Combination to dfRMSE
dfRMSE.loc['FC_cluster_DST'] = [rmse_fc_cluster_dst, smape_fc_cluster_dst]

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

dfRMSE.to_csv("./Results/Tables/3_4_rmse.csv")

print(dfRMSE)

### Create new dataframe with date, job_no, sDepVar, and predicted values ###
dfDataPred = dfData[
    ['date', 'job_no', sDepVar, 'predicted_ols', 'predicted_lag', 'predicted_lag_budget', 'predicted_fc',
     'predicted_cluster_fc']]

# Save to .parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")

########################################################################################################################

plt.close('all')
