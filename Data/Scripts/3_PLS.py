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
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# Load ./dfData.parquet
# sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)

### Predict sDepVar using OLS ###

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
print(results.summary2())
# Save results to LaTeX
ols = results.summary2().as_latex()
with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict sDepVar using OLS
dfDataFinishedTrain['predicted'] = results.predict(dfDataFinishedTrain[lIndepVar])
dfDataFinishedTest['predicted'] = results.predict(dfDataFinishedTest[lIndepVar])

# Plot the sum of predicted and actual sDepVar by date
dfDataFinishedTrain['date'] = pd.to_datetime(dfDataFinishedTrain['date'])
dfDataFinishedTest['date'] = pd.to_datetime(dfDataFinishedTest['date'])

dfDataFinishedTrain = dfDataFinishedTrain.sort_values(by='date')
dfDataFinishedTest = dfDataFinishedTest.sort_values(by='date')

dfDataFinishedTrain['sum'] = dfDataFinishedTrain.groupby('date')['total_contribution'].transform('sum')
dfDataFinishedTrain['sum_predicted'] = dfDataFinishedTrain.groupby('date')['predicted'].transform('sum')
dfDataFinishedTrain['sum_budget'] = dfDataFinishedTrain.groupby('date')['final_estimate_contribution'].transform('sum')

dfDataFinishedTest['sum'] = dfDataFinishedTest.groupby('date')['total_contribution'].transform('sum')
dfDataFinishedTest['sum_predicted'] = dfDataFinishedTest.groupby('date')['predicted'].transform('sum')
dfDataFinishedTest['sum_budget'] = dfDataFinishedTest.groupby('date')['final_estimate_contribution'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum'], label='Actual')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_predicted'], label='Predicted')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_budget'], label='Budget')
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
plt.show()


### Add lagged variables to lIndepVar ###
# Add lagged variables to lIndepVar
lIndepVar_lag = lIndepVar + ['contribution_lag1','revenue_lag1','costs_lag1',
                             'contribution_lag2','revenue_lag2','costs_lag2',
                             'contribution_lag3','revenue_lag3','costs_lag3']

# Run OLS with lagged variables
model = sm.OLS(dfDataFinishedTrain[sDepVar], dfDataFinishedTrain[lIndepVar_lag], missing='drop')
results = model.fit()
print(results.summary2())
# Save results to LaTeX
ols = results.summary2().as_latex()
with open('Results/Tables/3_2_ols_lag.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict sDepVar using OLS
dfDataFinishedTrain['predicted_lag'] = results.predict(dfDataFinishedTrain[lIndepVar_lag])
dfDataFinishedTest['predicted_lag'] = results.predict(dfDataFinishedTest[lIndepVar_lag])

dfDataFinishedTrain['sum_predicted_lag'] = dfDataFinishedTrain.groupby('date')['predicted_lag'].transform('sum')
dfDataFinishedTest['sum_predicted_lag'] = dfDataFinishedTest.groupby('date')['predicted'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum'], label='Actual')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_predicted_lag'], label='Predicted')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_budget'], label='Budget')
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


# Include production_estimate_contribution and sales_estimate_contribution
lIndepVar_lag_budget = lIndepVar_lag + ['production_estimate_contribution', 'sales_estimate_contribution']

# Run OLS with lagged variables and budget
model = sm.OLS(dfDataFinishedTrain[sDepVar], dfDataFinishedTrain[lIndepVar_lag_budget], missing='drop')
results = model.fit()
print(results.summary2())
# Save results to LaTeX
ols = results.summary2().as_latex()
with open('Results/Tables/3_3_ols_lag_budget.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict sDepVar using OLS
dfDataFinishedTrain['predicted_lag_budget'] = results.predict(dfDataFinishedTrain[lIndepVar_lag_budget])
dfDataFinishedTest['predicted_lag_budget'] = results.predict(dfDataFinishedTest[lIndepVar_lag_budget])

dfDataFinishedTrain['sum_predicted_lag_budget'] = dfDataFinishedTrain.groupby('date')['predicted_lag_budget'].transform('sum')
dfDataFinishedTest['sum_predicted_lag_budget'] = dfDataFinishedTest.groupby('date')['predicted_lag_budget'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum'], label='Actual')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_predicted_lag_budget'], label='Predicted')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_budget'], label='Budget')
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

### Predict sDepVar using PLS ###
# Run PLS
pls = PLSRegression(n_components=5)
pls.fit(scaled_train[lIndepVar_lag_budget], dfDataFinishedTrain[sDepVar])

# Cross-validation
y_cv = cross_val_predict(pls, scaled_train[lIndepVar_lag_budget], dfDataFinishedTrain[sDepVar], cv=10)

# Calculate RMSE and R^2
rmse = np.sqrt(mean_squared_error(dfDataFinishedTrain[sDepVar], y_cv))


# Save results to LaTeX
pls = DataFrame(pls.coef_, index=lIndepVar_lag_budget, columns=['PLS'])
with open('Results/Tables/3_4_pls.tex', 'w', encoding='utf-8') as f:
    f.write(pls.to_latex())

# Predict sDepVar using PLS
dfDataFinishedTrain['predicted_pls'] = pls.predict(scaled_train[lIndepVar_lag_budget])
dfDataFinishedTest['predicted_pls'] = pls.predict(scaled_test[lIndepVar_lag_budget])

# Rescale predicted values
dfDataFinishedTrain['predicted_pls'] = scaler.inverse_transform(dfDataFinishedTrain['predicted_pls'])
dfDataFinishedTest['predicted_pls'] = scaler.inverse_transform(dfDataFinishedTest['predicted_pls'])

dfDataFinishedTrain['sum_predicted_pls'] = dfDataFinishedTrain.groupby('date')['predicted_pls'].transform('sum')
dfDataFinishedTest['sum_predicted_pls'] = dfDataFinishedTest.groupby('date')['predicted_pls'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum'], label='Actual')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_predicted_pls'], label='Predicted')
ax.plot(dfDataFinishedTrain['date'], dfDataFinishedTrain['sum_budget'], label='Budget')
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
plt.savefig("./Results/Figures/3_4_pls.png")
plt.savefig("./Results/Presentation/3_4_pls.svg")
plt.show()

