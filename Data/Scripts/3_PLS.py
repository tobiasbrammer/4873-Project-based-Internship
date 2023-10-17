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

# Load data
dfDataScaled = pd.read_parquet("./dfData_reg_scaled.parquet")
dfData = pd.read_parquet("./dfData_reg.parquet")

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

# Rescale dfDataScaled to dfData
dfDataRescaled = dfDataScaled.copy()
dfDataRescaled[colIndepVarNum] = x_scaler.inverse_transform(dfDataScaled[colIndepVarNum].values)
dfDataRescaled[sDepVar] = y_scaler.inverse_transform(dfDataScaled[sDepVar].values.reshape(-1, 1))

### Predict sDepVar using OLS ###
# Get the 5 most correlated variables (of numeric variables)
lNumericCols = dfDataScaled.select_dtypes(include=[np.number]).columns.tolist()

### Split dfDataScaled into train and test ###
# Get index of train from dfData['train']
train_index = dfData[dfData['train'] == 1].index

# Split dfDataScaled into train and test
dfDataScaledTrain = dfDataScaled.loc[train_index]
dfDataScaledTest = dfDataScaled.drop(train_index)

corr = dfData[dfData['train'] == 1][lNumericCols].corr()
corr = corr.sort_values(by=sDepVar, ascending=False)
corr = corr[sDepVar]
# Filter out variables with "contribution" or "revenue" in the name
corr = corr[~corr.index.str.contains('contribution|revenue|costs')]
corr = corr[0:10]
# Save the 5 most correlated variables in a list
lIndepVar = corr.index.tolist()

# Plot correlation between sDepVar and lIndepVar
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(dfData[dfData['train'] == 1][[sDepVar] + lIndepVar].corr(), annot=True, fmt='.2f',
            cmap=sns.light_palette(vColors[0], as_cmap=True))
plt.title(f'Correlation between {sDepVar} and selected variables')
plt.savefig("./Results/Figures/3_0_corr.png")
plt.savefig("./Results/Presentation/3_0_corr.svg")

# Run OLS
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar])
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05, slim=True).as_latex()

with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict and rescale sDepVar using OLS
dfData['predicted_ols'] = results.predict(dfDataScaled[lIndepVar])

dfData['predicted_ols'] = y_scaler.inverse_transform(dfData['predicted_ols'].values.reshape(-1, 1))

# Group by date and sum predicted_ols
dfData['sum_predicted_ols'] = dfData.groupby('date')['predicted_ols'].transform('sum')
# Group by date and sum sDepVar
dfData['sum'] = dfData.groupby('date')[sDepVar].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_ols'], label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Total Contribution')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.15),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/3_1_ols.png")
plt.savefig("./Results/Presentation/3_1_ols.svg")

# Calculate out-of-sample RMSE of OLS
rmse_ols = np.sqrt(
    mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_ols']))
# symmetric Mean Absolute Error (sMAPE)
smape_ols = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_ols']) /
                    (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                        dfData[dfData['train'] == 0]['predicted_ols']))) * 100

### Add lagged variables to lIndepVar ###
# Add lagged variables to lIndepVar
lIndepVar_lag = lIndepVar + ['contribution_lag1', 'revenue_lag1', 'costs_lag1',
                             'contribution_lag2', 'revenue_lag2', 'costs_lag2',
                             'contribution_lag3', 'revenue_lag3', 'costs_lag3']

# Correlation between sDepVar and lIndepVar_lag
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(dfData[dfData['train'] == 1][[sDepVar] + lIndepVar_lag].corr(), annot=True, fmt='.2f',
            cmap=sns.light_palette(vColors[0], as_cmap=True))
plt.title(f'Correlation between {sDepVar} and selected variables and lags')
plt.savefig("./Results/Figures/3_1_corr_incl_lag.png")
plt.savefig("./Results/Presentation/3_1_corr_incl_lag.svg")

# Run OLS with lagged variables
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar_lag], missing='drop')
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05, slim=True).as_latex()

with open('Results/Tables/3_2_ols_lag.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict and rescale sDepVar using OLS with lagged variables
dfData['predicted_lag'] = results.predict(dfDataScaled[lIndepVar_lag])

dfData['predicted_lag'] = y_scaler.inverse_transform(dfData['predicted_lag'].values.reshape(-1, 1))

# Group by date and sum predicted_lag
dfData['sum_predicted_lag'] = dfData.groupby('date')['predicted_lag'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_lag'], label='Predicted (incl. lag)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Total Contribution')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
                xy=(1.0, -0.15),
                color='grey',
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)
plt.savefig("./Results/Figures/3_2_ols_lag.png")
plt.savefig("./Results/Presentation/3_2_ols_lag.svg")

# Calculate RMSE of OLS with lagged variables
rmse_ols_lag = np.sqrt(mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_lag']))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_lag']) /
                    (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                        dfData[dfData['train'] == 0]['predicted_lag']))) * 100


# Include production_estimate_contribution and sales_estimate_contribution
lIndepVar_lag_budget = lIndepVar_lag + ['production_estimate_contribution', 'sales_estimate_contribution']

# Save lIndepVar_lag_budget to .AUX/
with open('./.AUX/lIndepVar_lag_budget.txt', 'w') as lVars:
    lVars.write('\n'.join(lIndepVar_lag_budget))


# Correlation between sDepVar and lIndepVar_lag_budget
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(dfData[dfData['train'] == 1][[sDepVar] + lIndepVar_lag_budget].corr(), annot=True, fmt='.2f',
            cmap=sns.light_palette(vColors[0], as_cmap=True))
plt.title(f'Correlation between {sDepVar} and selected variables, lags and budget')
plt.savefig("./Results/Figures/3_2_corr_incl_lag_budget.png")
plt.savefig("./Results/Presentation/3_2_corr_incl_lag_budget.svg")

# Run OLS with lagged variables and budget
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar_lag_budget])
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05, slim=True).as_latex()

with open('Results/Tables/3_3_ols_lag_budget.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict and rescale sDepVar using OLS with lagged variables and budget
dfData['predicted_lag_budget'] = results.predict(dfDataScaled[lIndepVar_lag_budget])

dfData['predicted_lag_budget'] = y_scaler.inverse_transform(dfData['predicted_lag_budget'].values.reshape(-1, 1))

# Group by date and sum predicted_lag_budget
dfData['sum_predicted_lag_budget'] = dfData.groupby('date')['predicted_lag_budget'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_lag_budget'], label='Predicted (incl. lag and budget)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Total Contribution')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
                xy=(1.0, -0.15),
                color='grey',
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)
plt.savefig("./Results/Figures/3_3_ols_lag_budget.png")
plt.savefig("./Results/Presentation/3_3_ols_lag_budget.svg")

# Calculate RMSE of OLS with lagged variables and budget
rmse_ols_lag_budget = np.sqrt(mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_lag_budget']))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag_budget = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_lag_budget']) /
                    (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                        dfData[dfData['train'] == 0]['predicted_lag_budget']))) * 100


### Predict sDepVar using PLS ###
# Run PLS
pls = PLSRegression(n_components=20, scale=False, max_iter=5000)
pls.fit(dfDataScaledTrain[lIndepVar_lag_budget], dfDataScaledTrain[sDepVar])

# Predict and rescale sDepVar using PLS
dfData['predicted_pls'] = pls.predict(dfDataScaled[lIndepVar_lag_budget])

dfData['predicted_pls'] = y_scaler.inverse_transform(dfData['predicted_pls'].values.reshape(-1, 1))

# Group by date and sum predicted_pls
dfData['sum_predicted_pls'] = dfData.groupby('date')['predicted_pls'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_pls'], label='Predicted (PLS)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Total Contribution')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
                xy=(1.0, -0.15),
                color='grey',
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)
plt.savefig("./Results/Figures/3_4_pls.png")
plt.savefig("./Results/Presentation/3_4_pls.svg")
plt.show()

# Calculate RMSE of PLS
rmse_pls = np.sqrt(mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_pls']))
# symmetric Mean Absolute Error (sMAPE)
smape_pls = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_pls']) /
                    (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                        dfData[dfData['train'] == 0]['predicted_pls']))) * 100

### Forecast Combination ###
# Produce a combined forecast of ols_lag_budget and pls
dfData['predicted_fc'] = (dfData['predicted_ols'] + dfData['predicted_pls']) / 2

# Group by date and sum predicted_fc
dfData['sum_predicted_fc'] = dfData.groupby('date')['predicted_fc'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_fc'], label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Total Contribution')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
                xy=(1.0, -0.15),
                color='grey',
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)
plt.savefig("./Results/Figures/3_4_fc.png")
plt.savefig("./Results/Presentation/3_4_fc.svg")
plt.show()

# Calculate RMSE of Forecast Combination
rmse_fc = np.sqrt(mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_fc']))
# symmetric Mean Absolute Error (sMAPE)
smape_fc = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_fc']) /
                    (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                        dfData[dfData['train'] == 0]['predicted_fc']))) * 100



# Compare RMSE and sMAPE of the different models in a table
dfRMSE = pd.DataFrame({'RMSE': [rmse_ols, rmse_ols_lag, rmse_ols_lag_budget, rmse_pls, rmse_fc],
                          'sMAPE': [smape_ols, smape_ols_lag, smape_ols_lag_budget, smape_pls, smape_fc]},
                            index=['OLS', 'OLS with lagged variables', 'OLS with lagged variables and budget', 'PLS', 'FC'])

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

# Bold the lowest RMSE
dfRMSE.loc[dfRMSE['RMSE'] == dfRMSE['RMSE'].min(), 'RMSE'] = '\\textbf{' + dfRMSE['RMSE'].astype(str) + '}'

# Bold the lowest sMAPE
dfRMSE.loc[dfRMSE['sMAPE'] == dfRMSE['sMAPE'].min(), 'sMAPE'] = '\\textbf{' + dfRMSE['sMAPE'].astype(str) + '}'

print(dfRMSE)

# Output to LaTeX
dfRMSE = dfRMSE.style.to_latex(
    caption='RMSE of Naive Methods',
    position_float='centering',
    position='h!',
    hrules=True,
    label='naive_rmse')

# Output to LaTeX with encoding
with open('Results/Tables/3_4_rmse.tex', 'w', encoding='utf-8') as f:
    f.write(dfRMSE)

### Get Prediction of job_no S161210 ###
# Get the data of job_no S161210
sJobNo = 'S161210'
dfDataJob = dfData[dfData['job_no'] == sJobNo]

# Plot the actual and predicted contribution of job_no S161210
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataJob['date'], dfDataJob['contribution'], label='Actual')
ax.plot(dfDataJob['date'], dfDataJob['predicted_fc'], label='Predicted (Forecast Combination)')
ax.plot(dfDataJob['date'], dfDataJob['predicted_pls'], label='Predicted (PLS)')
ax.plot(dfDataJob['date'], dfDataJob['predicted_ols'], label='Predicted (OLS)')
ax.set_xlabel('Date')
ax.set_ylabel('Contribution')
ax.set_title(f'Actual vs. Predicted Contribution of {sJobNo}')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
                xy=(1.0, -0.15),
                color='grey',
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)
plt.show()

# Save dfData
dfData.to_parquet("./dfData_reg.parquet")

# Save dfRMSE to Results/Tables/3_4_rmse.csv
dfRMSE.to_csv("./Results/Tables/3_4_rmse.csv")
