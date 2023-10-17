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

# Import sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

### Predict sDepVar using OLS ###
# Get the 5 most correlated variables (of numeric variables)
lNumericCols = dfDataTrain.select_dtypes(include=[np.number]).columns.tolist()
corr = dfDataTrain[lNumericCols].corr()
corr = corr.sort_values(by=sDepVar, ascending=False)
corr = corr[sDepVar]
# Filter out variables with "contribution" or "revenue" in the name
corr = corr[~corr.index.str.contains('contribution|revenue|costs')]
corr = corr[0:10]
# Save the 5 most correlated variables in a list
lIndepVar = corr.index.tolist()

# Run OLS
model = sm.OLS(dfDataTrain[sDepVar], dfDataTrain[lIndepVar])
results = model.fit()
print(results.summary2())
# Save results to LaTeX
ols = results.summary2().as_latex()
with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict sDepVar using OLS

# Rescale predicted values
dfDataTrain['predicted_ols'] = results.predict(dfDataTrain[lIndepVar])
dfDataTest['predicted_ols'] = results.predict(dfDataTest[lIndepVar])

dfDataTrain['predicted_ols'] = y_scaler.inverse_transform(dfDataTrain['predicted_ols'].values.reshape(-1, 1))
dfDataTest['predicted_ols'] = y_scaler.inverse_transform(dfDataTest['predicted_ols'].values.reshape(-1, 1))

# Plot the sum of predicted and actual sDepVar by date
dfDataTrain['date'] = pd.to_datetime(dfDataTrain['date'])
dfDataTest['date'] = pd.to_datetime(dfDataTest['date'])

dfDataTrain = dfDataTrain.sort_values(by='date')
dfDataTest = dfDataTest.sort_values(by='date')

dfDataTrain['sum'] = dfDataTrain.groupby('date')['total_contribution'].transform('sum')
dfDataTrain['sum_predicted'] = dfDataTrain.groupby('date')['predicted_ols'].transform('sum')
dfDataTrain['final_estimate'] = dfDataTrain.groupby('date')['final_estimate_contribution'].transform('sum')

dfDataTest['sum'] = dfDataTest.groupby('date')['total_contribution'].transform('sum')
dfDataTest['sum_predicted'] = dfDataTest.groupby('date')['predicted_ols'].transform('sum')
dfDataTest['final_estimate'] = dfDataTest.groupby('date')['final_estimate_contribution'].transform('sum')


# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataTest['date'], dfDataTest['sum'], label='Actual')
ax.plot(dfDataTest['date'], dfDataTest['sum_predicted'], label='Predicted')
ax.plot(dfDataTest['date'], dfDataTest['final_estimate'], label='Budget')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Contribution')
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
plt.show()

# Calculate RMSE of OLS and budget
rmse_ols = np.sqrt(mean_squared_error(dfDataTest['sum'], dfDataTest['sum_predicted']))
rmse_budget = np.sqrt(mean_squared_error(dfDataTest['sum'], dfDataTest['final_estimate_contribution']))
rmse_production_estimate = np.sqrt(mean_squared_error(dfDataTest['sum'], dfDataTest['production_estimate_contribution']))


### Add lagged variables to lIndepVar ###
# Add lagged variables to lIndepVar
lIndepVar_lag = lIndepVar + ['contribution_lag1','revenue_lag1','costs_lag1',
                             'contribution_lag2','revenue_lag2','costs_lag2',
                             'contribution_lag3','revenue_lag3','costs_lag3']

# Run OLS with lagged variables
model = sm.OLS(dfDataTrain[sDepVar], dfDataTrain[lIndepVar_lag], missing='drop')
results = model.fit()
print(results.summary2())
# Save results to LaTeX
ols = results.summary2().as_latex()
with open('Results/Tables/3_2_ols_lag.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict sDepVar using OLS
dfDataTrain['predicted_lag'] = results.predict(dfDataTrain[lIndepVar_lag])
dfDataTest['predicted_lag'] = results.predict(dfDataTest[lIndepVar_lag])
# Rescale predicted values
dfDataTrain['predicted_lag'] = y_scaler.inverse_transform(dfDataTrain['predicted_lag'].values.reshape(-1, 1))
dfDataTest['predicted_lag'] = y_scaler.inverse_transform(dfDataTest['predicted_lag'].values.reshape(-1, 1))


dfDataTrain['sum_predicted_lag'] = dfDataTrain.groupby('date')['predicted_lag'].transform('sum')
dfDataTest['sum_predicted_lag'] = dfDataTest.groupby('date')['predicted_lag'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataTest['date'], dfDataTest['sum'], label='Actual')
ax.plot(dfDataTest['date'], dfDataTest['sum_predicted_lag'], label='Predicted')
ax.plot(dfDataTest['date'], dfDataTest['final_estimate'], label='Budget')
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
plt.savefig("./Results/Figures/3_2_ols_lag.png")
plt.savefig("./Results/Presentation/3_2_ols_lag.svg")
plt.show()

# Calculate RMSE of OLS with lagged variables and budget
rmse_ols_lag = np.sqrt(mean_squared_error(dfDataTest['sum'], dfDataTest['sum_predicted_lag']))

# Include production_estimate_contribution and sales_estimate_contribution
lIndepVar_lag_budget = lIndepVar_lag + ['production_estimate_contribution', 'sales_estimate_contribution']

# Run OLS with lagged variables and budget
model = sm.OLS(dfDataTrain[sDepVar], dfDataTrain[lIndepVar_lag_budget], missing='drop')
results = model.fit()
print(results.summary2())
# Save results to LaTeX
ols = results.summary2().as_latex()
with open('Results/Tables/3_3_ols_lag_budget.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict sDepVar using OLS
dfDataTrain['predicted_lag_budget'] = results.predict(dfDataTrain[lIndepVar_lag_budget])
dfDataTest['predicted_lag_budget'] = results.predict(dfDataTest[lIndepVar_lag_budget])

# Rescale predicted values
dfDataTrain['predicted_lag_budget'] = y_scaler.inverse_transform(dfDataTrain['predicted_lag_budget'].values.reshape(-1, 1))
dfDataTest['predicted_lag_budget'] = y_scaler.inverse_transform(dfDataTest['predicted_lag_budget'].values.reshape(-1, 1))

dfDataTrain['sum_predicted_lag_budget'] = dfDataTrain.groupby('date')['predicted_lag_budget'].transform('sum')
dfDataTest['sum_predicted_lag_budget'] = dfDataTest.groupby('date')['predicted_lag_budget'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataTest['date'], dfDataTest['sum'], label='Actual')
ax.plot(dfDataTest['date'], dfDataTest['sum_predicted_lag_budget'], label='Predicted')
ax.plot(dfDataTest['date'], dfDataTest['final_estimate'], label='Budget')
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
plt.savefig("./Results/Figures/3_3_ols_lag_budget.png")
plt.savefig("./Results/Presentation/3_3_ols_lag_budget.svg")
plt.show()

# Calculate RMSE of OLS with lagged variables and budget
rmse_ols_lag_budget = np.sqrt(mean_squared_error(dfDataTest['sum'], dfDataTest['sum_predicted_lag_budget']))


### Predict sDepVar using PLS ###
# Run PLS
pls = PLSRegression(n_components=20, scale=False, max_iter=5000)
pls.fit(dfDataTrain[lIndepVar_lag_budget], dfDataTrain[sDepVar])

# Predict sDepVar using PLS
dfDataTrain['predicted_pls'] = pls.predict(dfDataTrain[lIndepVar_lag_budget])
dfDataTest['predicted_pls'] = pls.predict(dfDataTest[lIndepVar_lag_budget])
# Reshape predicted values
dfDataTrain['predicted_pls'] = dfDataTrain['predicted_pls'].values.reshape(-1, 1)
dfDataTest['predicted_pls'] = dfDataTest['predicted_pls'].values.reshape(-1, 1)
# Rescale predicted values
dfDataTrain['predicted_pls'] = y_scaler.inverse_transform(dfDataTrain['predicted_pls'].values.reshape(-1, 1))
dfDataTest['predicted_pls'] = y_scaler.inverse_transform(dfDataTest['predicted_pls'].values.reshape(-1, 1))

dfDataTrain['sum_predicted_pls'] = dfDataTrain.groupby('date')['predicted_pls'].transform('sum')
dfDataTest['sum_predicted_pls'] = dfDataTest.groupby('date')['predicted_pls'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataTest['date'], dfDataTest['sum'], label='Actual')
ax.plot(dfDataTest['date'], dfDataTest['sum_predicted_pls'], label='Predicted')
ax.plot(dfDataTest['date'], dfDataTest['final_estimate'], label='Budget')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Contribution')
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
plt.savefig("./Results/Figures/3_4_pls.png")
plt.savefig("./Results/Presentation/3_4_pls.svg")
plt.show()

# Calculate RMSE of PLS
rmse_pls = np.sqrt(mean_squared_error(dfDataTest['sum'], dfDataTest['sum_predicted_pls']))

# Compare rmse in a table
dfRMSE = pd.DataFrame({'RMSE': [rmse_ols, rmse_ols_lag, rmse_ols_lag_budget, rmse_pls, rmse_budget, rmse_production_estimate]},
                        index=['OLS', 'OLS with lagged variables', 'OLS with lagged variables and budget','PLS with lagged variables and budget', 'Final estimate', 'Production estimate'])

dfRMSE = dfRMSE.round(4).applymap('{:,.4f}'.format)

# Bold the lowest RMSE
dfRMSE.loc[dfRMSE['RMSE'] == dfRMSE['RMSE'].min(), 'RMSE'] = '\\textbf{' + dfRMSE['RMSE'].astype(str) + '}'


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
sJobNo = 'S333720'
dfDataJobTr = dfDataTrain[dfDataTrain['job_no'] == sJobNo]
dfDataJobTe = dfDataTest[dfDataTest['job_no'] == sJobNo]

# Append dfDataJobTr and dfDataJobTe
dfDataJob = pd.concat([dfDataJobTr, dfDataJobTe], axis=0)

dfDataJob = dfDataJob.sort_values(by='date')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataJob['date'], dfDataJob['contribution'], label='Actual')
ax.plot(dfDataJob['date'], dfDataJob['predicted_ols'], label='OLS')
ax.plot(dfDataJob['date'], dfDataJob['predicted_lag'], label='OLS with lagged variables')
ax.plot(dfDataJob['date'], dfDataJob['predicted_lag_budget'], label='OLS with lagged variables and budget')
ax.plot(dfDataJob['date'], dfDataJob['predicted_pls'], label='PLS')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Actual vs. Predicted Contribution')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.show()