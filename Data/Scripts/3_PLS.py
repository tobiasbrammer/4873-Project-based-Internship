# Import required libraries
import os
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import statsmodels.api as sm
import joblib
from plot_config import *
from predict_and_scale import *
from plot_predicted import *
from smape import *
from sklearn.metrics import mean_squared_error

pd.options.mode.chained_assignment = None  # default='warn'

# Load ./dfData.parquet
if os.name == 'posix':
    sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
# If operating system is Windows then
elif os.name == 'nt':
    sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)

# Load data
dfDataScaled = pd.read_parquet("./dfData_reg_scaled.parquet")
dfData = pd.read_parquet("./dfData_reg.parquet")
dfDataWIP = pd.read_parquet("./dfData_reg_scaled_wip.parquet")
dfDataOrg = pd.read_parquet("./dfData_org.parquet")

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
lDST = [s for s in dfDataScaled.columns if 'kbyg' in s]

# Write lDST to .AUX/
with open('./.AUX/lDST.txt', 'w') as lVars:
    lVars.write('\n'.join(lDST))

model = sm.OLS(dfDataScaledTrain[sDepVar], sm.add_constant(dfDataScaledTrain[lDST]))
results_dst = model.fit()
# Save results to LaTeX
ols = results_dst.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_0_dst.tex', 'w', encoding='utf-8') as f:
    f.write(ols)
upload(ols, 'Project-based Internship', 'tables/3_0_dst.tex')

### Create function to predict sequentially for each job ###
# By sequentially introducing unseen data to the model,
# the study evaluates the model's predictive accuracy as each project advances.


# Predict sDepVar using OLS for each job.
# Predict sDepVar for each job number by exposing the model to an expanding window of data. Start with the first observation and add one observation for each iteration.

# Create a list of job numbers
lJobNo = dfData['job_no'].unique().tolist()
lJobNoWIP = dfDataWIP['job_no'].unique().tolist()
# Omit nan from lJobNo
lJobNo = [x for x in lJobNo if str(x) != 'nan']
lJobNoWIP = [x for x in lJobNoWIP if str(x) != 'nan']

# Save lJobNo to .AUX/
with open('./.AUX/lJobNo.txt', 'w') as lVars:
    lVars.write('\n'.join(lJobNo))

# Save lJobNoWIP to .AUX/
with open('./.AUX/lJobNoWIP.txt', 'w') as lVars:
    lVars.write('\n'.join(lJobNoWIP))


predict_and_scale(dfData, dfDataScaled, results_dst, 'dst', lDST, lJobNo)

dfDataPred = dfData[['date', 'job_no', sDepVar, 'predicted_dst']]

plot_predicted(dfData, 'predicted_dst', 'Statistics Denmark', '3_0_dst', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar, show=False)

# Calculate out-of-sample RMSE of DST
rmse_dst = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_dst'].replace(np.nan, 0)
                       ))
# symmetric Mean Absolute Error (sMAPE)
smape_dst = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_dst'])

# Predict and scale dfDataWIP
predict_and_scale(dfDataWIP, dfDataWIP, results_dst, 'dst', lDST, lJobNoWIP)

### OLS with s-curve differences. ###
lSCurve = ['revenue_scurve_diff', 'costs_scurve_diff', 'contribution_scurve_diff']

model = sm.OLS(dfDataScaledTrain[sDepVar], sm.add_constant(dfDataScaledTrain[lSCurve]))
results_scurve = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
# Save model to .MODS/
results_scurve.save('./.MODS/results_scurve.pickle')
# Save results to LaTeX
ols = results_scurve.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_9_scurve.tex', 'w', encoding='utf-8') as f:
    f.write(ols)
upload(ols, 'Project-based Internship', 'tables/3_9_scurve.tex')

# Predict and rescale sDepVar using OLS
predict_and_scale(dfData, dfDataScaled, results_scurve, 'scurve', lSCurve, lJobNo)

dfDataPred['predicted_scurve'] = dfData['predicted_scurve']

plot_predicted(dfData, 'predicted_scurve', 'S-curve', '3_9_scurve', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate out-of-sample RMSE of DST
rmse_scurve = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_scurve'].replace(np.nan, 0)
                       ))
# symmetric Mean Absolute Error (sMAPE)
smape_scurve = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_scurve'].replace(np.nan, 0))

# Predict dfDataWIP[sDepVar]
predict_and_scale(dfDataWIP, dfDataWIP, results_scurve, 'scurve', lSCurve, lJobNoWIP)

### Using correlation to select variables ###
corr = dfData[dfData[trainMethod] == 1][lNumericCols].corr()
corr = corr.sort_values(by=sDepVar, ascending=False)
corr = corr[sDepVar]
# Filter out variables with "contribution" or "revenue" in the name
corr = corr[~corr.index.str.contains('contribution')]
corr = corr[~corr.index.str.contains('lag')]
corr = corr[~corr.index.str.contains('cluster')]
corr = corr[~corr.index.str.contains('revenue')]
corr = corr[~corr.index.str.contains('estimate')]
corr = corr[~corr.index.str.contains('budget')]
corr = corr[~corr.index.str.contains('days')]
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
model = sm.OLS(dfDataScaledTrain[sDepVar], sm.add_constant(dfDataScaledTrain[lIndepVar]), missing='drop')
results_ols = model.fit()
# Save results to LaTeX
ols = results_ols.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)
upload(ols, 'Project-based Internship', 'tables/3_1_ols.tex')

# Predict and rescale sDepVar using OLS with lIndepVar
predict_and_scale(dfData, dfDataScaled, results_ols, 'ols', lIndepVar, lJobNo)

dfDataPred['predicted_ols'] = dfData['predicted_ols']

# Add production_estimate_contribution to dfDataPred
dfDataPred['production_estimate_contribution'] = dfData['production_estimate_contribution']
# Add final_estimate_contribution to dfDataPred
dfDataPred['final_estimate_contribution'] = dfData['final_estimate_contribution']
dfDataPred['risk'] = dfData['risk']

plot_predicted(dfData, 'predicted_ols', 'OLS', '3_1_ols', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate out-of-sample RMSE of OLS
rmse_ols = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_ols'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_ols'])

# Predict dfDataWIP[sDepVar]
predict_and_scale(dfDataWIP, dfDataWIP, results_ols, 'ols', lIndepVar, lJobNoWIP)

### Add lagged variables to lIndepVar ###
# lIndepVar_lag = lIndepVar + ['contribution_lag1', 'revenue_lag1', 'costs_lag1',
#                              'contribution_lag2', 'revenue_lag2', 'costs_lag2',
#                              'contribution_lag3', 'revenue_lag3', 'costs_lag3']

lIndepVar_lag = lIndepVar + ['revenue_lag1', 'costs_lag1', 'labor_cost_share_lag1']

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
model = sm.OLS(dfDataScaledTrain[sDepVar], sm.add_constant(dfDataScaledTrain[lIndepVar_lag]), missing='drop')
results_ols_lag = model.fit()
# Save results to LaTeX
ols = results_ols_lag.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_2_ols_lag.tex', 'w', encoding='utf-8') as f:
    f.write(ols)
upload(results_ols_lag.summary().as_latex(), 'Project-based Internship', 'tables/3_2_ols_lag.tex')

# Predict and rescale sDepVar using OLS with lagged variables
predict_and_scale(dfData, dfDataScaled, results_ols_lag, 'ols_lag', lIndepVar_lag, lJobNo)
dfDataPred['predicted_lag'] = dfData['predicted_lag']
plot_predicted(dfData, 'predicted_lag', 'OLS with lag', '3_2_ols_lag', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)


# Calculate RMSE of OLS with lagged variables
rmse_ols_lag = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_lag'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                      dfData[dfData[trainMethod] == 0]['predicted_lag'])

# Predict dfDataWIP[sDepVar]
predict_and_scale(dfDataWIP, dfDataWIP, results_ols_lag, 'ols_lag', lIndepVar_lag, lJobNoWIP)

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
model = sm.OLS(dfDataScaledTrain[sDepVar], sm.add_constant(dfDataScaledTrain[lIndepVar_lag_budget]), missing='drop')
results_lag_budget = model.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
# Save model to .MODS/
results_lag_budget.save('./.MODS/results_lag_budget.pickle')

# Save results to LaTeX
ols = results_lag_budget.summary(alpha=0.05).as_latex()

# Check for multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif_data = pd.DataFrame()
vif_data["feature"] = dfDataScaledTrain[lIndepVar_lag_budget].columns

# Calculate VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(dfDataScaledTrain[lIndepVar_lag_budget].values, i)
                     for i in range(len(dfDataScaledTrain[lIndepVar_lag_budget].columns))]
print(vif_data)


with open('Results/Tables/3_3_ols_lag_budget.tex', 'w', encoding='utf-8') as f:
    f.write(ols)
upload(ols, 'Project-based Internship', 'tables/3_3_ols_lag_budget.tex')

# Predict and rescale sDepVar using OLS with lagged variables and budget
predict_and_scale(dfData, dfDataScaled, results_lag_budget, 'ols_lag_budget', lIndepVar_lag_budget, lJobNo)
dfDataPred['predicted_lag_budget'] = dfData['predicted_lag_budget']
plot_predicted(dfData, 'predicted_lag_budget', 'OLS with lag and budget', '3_3_ols_lag_budget', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of OLS with lagged variables and budget
rmse_ols_lag_budget = np.sqrt(mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                                                 dfData[dfData[trainMethod] == 0]['predicted_lag_budget'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag_budget = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                             dfData[dfData[trainMethod] == 0]['predicted_lag_budget'])

# Predict dfDataWIP[sDepVar]
predict_and_scale(dfDataWIP, dfDataWIP, results_lag_budget, 'ols_lag_budget', lIndepVar_lag_budget, lJobNoWIP)

### Forecast Combination ###
# Produce a combined forecast of ols_lag_budget and pls
dfData['predicted_fc'] = (dfData['predicted_dst'] + dfData['predicted_lag_budget']) / 2
dfDataPred['predicted_fc'] = dfData['predicted_fc']

plot_predicted(dfData, 'predicted_fc', 'OLS Forecast Combination', '3_5_fc', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)


# Calculate RMSE of Forecast Combination
rmse_fc = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_fc'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_fc'])

# Compare RMSE and sMAPE of the different models in a table
rmse_final = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['final_estimate_contribution'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_final = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['final_estimate_contribution'])

rmse_prod = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['production_estimate_contribution'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_prod = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['production_estimate_contribution'])


dfRMSE = pd.DataFrame({'RMSE': [rmse_final, rmse_prod, rmse_dst, rmse_scurve, rmse_ols, rmse_ols_lag, rmse_ols_lag_budget, rmse_fc],
                       'sMAPE': [smape_final, smape_prod, smape_dst, smape_scurve, smape_ols, smape_ols_lag, smape_ols_lag_budget, smape_fc]},
                      index=['Final Estimate', 'Production Estimate', 'DST', 'S-curve', 'OLS', 'OLS with lagged variables', 'OLS with lags and budget', 'OLS Forecast Combination'])
# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

### Use clustering to find similar jobs and predict sDepVar for each cluster ###
lCluster = [2, 3, 4, 5]
# For each cluster in cluster_{lCluster} do
for iCluster in lCluster:
    # Get the cluster labels to list using value_counts()
    lClusterLabels = dfData['cluster_' + str(iCluster)].value_counts().index.tolist()
    # For each cluster label in lClusterLabels do
    for iClusterLabel in lClusterLabels:
        # Filter the data based on cluster and label
        data_subset = dfDataScaledTrain[(dfDataScaledTrain['cluster_' + str(iCluster)] == iClusterLabel)]
        # Check if the subset of data has enough observations for OLS
        if data_subset.shape[0] > 1:
            # Run OLS
            model_cluster = sm.OLS(data_subset[sDepVar], sm.add_constant(data_subset[['intercept'] + lIndepVar_lag_budget]))
            results_cluster = model_cluster.fit(cov_type='HAC', cov_kwds={'maxlags': 3})
            # Save model to .MODS/
            results_cluster.save('./.MODS/results_cluster_' + str(iCluster) + '_' + str(iClusterLabel) + '.pickle')
            # Predict and rescale sDepVar using OLS with lagged variables and budget and add to cluster_{iCluster}
            dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
                iCluster)] = results_cluster.predict(
                dfDataScaled[dfDataScaled['cluster_' + str(iCluster)] == iClusterLabel][['intercept'] +lIndepVar_lag_budget]) # 22 cols
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
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual', linestyle='dashed')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_2'].transform('sum'),
        label='Predicted (2 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_3'].transform('sum'),
        label='Predicted (3 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_4'].transform('sum'),
        label='Predicted (4 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_5'].transform('sum'),
        label='Predicted (5 clusters)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.set_aspect('auto')
ax.set_ylim([-5, 20.00])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_6_cluster.png")
plt.savefig("./Results/Presentation/3_6_cluster.svg")
upload(plt, 'Project-based Internship', 'figures/3_6_cluster.png')
plt.close('all')

# Full sample
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual', linestyle='dashed')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_cluster_2'].transform('sum'),
        label='Predicted (2 clusters)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_cluster_3'].transform('sum'),
        label='Predicted (3 clusters)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_cluster_4'].transform('sum'),
        label='Predicted (4 clusters)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_cluster_5'].transform('sum'),
        label='Predicted (5 clusters)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.set_aspect('auto')
ax.set_ylim([-20, 100.00])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/FullSample/3_6_cluster_fs.png")
plt.savefig("./Results/Presentation/FullSample/3_6_cluster_fs.svg")
upload(plt, 'Project-based Internship', 'figures/FullSample/3_6_cluster_fs.png')
plt.close('all')


# Use Forecast Combination to combine the predictions of each cluster
# For each cluster in cluster_{lCluster} do
dfData['predicted_cluster_fc'] = (dfData['predicted_cluster_' + str(lCluster[0])].replace(np.nan, 0)
                                  + dfData['predicted_cluster_' + str(lCluster[1])].replace(np.nan, 0)
                                  + dfData['predicted_cluster_' + str(lCluster[2])].replace(np.nan, 0)
                                  + dfData['predicted_cluster_' + str(lCluster[3])].replace(np.nan, 0)) / 4

dfDataPred['predicted_cluster_fc'] = dfData['predicted_cluster_fc']

plot_predicted(dfData, 'predicted_cluster_fc', 'Cluster Combination', '3_7_fc_cluster',
               transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)


# Calculate RMSE of Forecast Combination
rmse_fc_cluster = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_cluster_fc'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc_cluster = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                         dfData[dfData[trainMethod] == 0]['predicted_cluster_fc'])

# Add RMSE and sMAPE of Forecast Combination to dfRMSE
dfRMSE.loc['Cluster Combination'] = [rmse_fc_cluster, smape_fc_cluster]

### Combine Cluster Forecast Combination and DST ###
dfData['predicted_fc_cluster_dst'] = (dfData['predicted_cluster_fc'] + dfData['predicted_dst']) / 2
dfDataPred['predicted_fc_cluster_dst'] = dfData['predicted_fc_cluster_dst']

# Calculate RMSE of Forecast Combination
rmse_fc_cluster_dst = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_fc_cluster_dst'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc_cluster_dst = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                             dfData[dfData[trainMethod] == 0]['predicted_fc_cluster_dst'])

# Add RMSE and sMAPE of Forecast Combination to dfRMSE
dfRMSE.loc['DST Cluster Combination'] = [rmse_fc_cluster_dst, smape_fc_cluster_dst]

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

dfRMSE.to_csv("./Results/Tables/3_4_rmse.csv")

print(dfRMSE)
########################################################################################################################


# Plot the sum of all predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual', linestyle='dashed')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['sales_estimate_contribution'].transform('sum'),
        label='Sales Estimate')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['production_estimate_contribution'].transform('sum'),
        label='Production Estimate')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['final_estimate_contribution'].transform('sum'),
        label='Final Estimate')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution (mDKK)')
ax.set_title('Out of Sample')
ax.set_aspect('auto')
ax.set_ylim([-5, 20.00])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_0_budget.png")
plt.savefig("./Results/Presentation/3_0_budget.svg")
upload(plt, 'Project-based Internship', 'figures/3_0_budget.png')
plt.close('all')

# Full sample budgets
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual', linestyle='dashed')
ax.plot(dfData['date'],
        dfData.groupby('date')['sales_estimate_contribution'].transform('sum'),
        label='Sales Estimate')
ax.plot(dfData['date'],
        dfData.groupby('date')['production_estimate_contribution'].transform('sum'),
        label='Production Estimate')
ax.plot(dfData['date'],
        dfData.groupby('date')['final_estimate_contribution'].transform('sum'),
        label='Final Estimate')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution (mDKK)')
ax.set_title('Full Sample')
ax.set_aspect('auto')
ax.set_ylim([-20, 100.00])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/FullSample/3_0_budget_fs.png")
plt.savefig("./Results/Presentation/FullSample/3_0_budget_fs.svg")
upload(plt, 'Project-based Internship', 'figures/FullSample/3_0_budget_fs.png')

########################################################################################################################

# Save to .parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")
dfDataWIP.to_parquet("./dfDataWIP_pred.parquet")

########################################################################################################################

plt.close('all')

########################################################################################################################

dfDesc = pd.read_parquet('./.AUX/dfDesc.parquet')
dfData_org = pd.read_parquet('./dfData_org.parquet')

lJob = ['S218705', 'S100762', 'S289834', 'S102941']

plt.close('all')

# Create a subplot for each job_no in lJob
fig, ax = plt.subplots(len(lJob), 1, figsize=(20, 10*len(lJob)))
# Loop through each job_no in lJob
for i, sJobNo in enumerate(lJob):
    # Plot total contribution, contribution, revenue and cumulative contribution
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_dst'],
               label='predicted (dst)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_scurve'],
               label='predicted (s-curve)', linestyle='dotted')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_ols'],
               label='predicted (ols)', linestyle='dashdot')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_fc_cluster_dst'],
               label='predicted (cluster)', linestyle='dashed')
    ax[i].plot(dfData[dfData['job_no'] == sJobNo]['date'],
               dfData_org[dfData_org['job_no'] == sJobNo]['contribution_cumsum'],
               label='cumulative contribution')
    ax[i].plot(dfData[dfData['job_no'] == sJobNo]['date'],
               dfData_org[dfData_org['job_no'] == sJobNo]['final_estimate_contribution'],
               label='slutvurdering')
    ax[i].axhline(y=0, color='black', linestyle='-')
    ax[i].set_xlabel('Date')
    ax[i].set_ylabel('Contribution')
    ax[i].set_title(f'Contribution of {sJobNo} - {dfDesc[dfDesc["job_no"] == sJobNo]["description"].values[0]}')
    ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.5)
    plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/Jobs/ols.png")
