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
    sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/PBI"
# If operating system is Windows then
elif os.name == 'nt':
    sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)

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
    token = subprocess.run(
        "curl https://api.dropbox.com/oauth2/token -d grant_type=refresh_token -d refresh_token=eztXuoP098wAAAAAAAAAAV4Ef4mnx_QpRaiqNX-9ijTuBKnX9LATsIZDPxLQu9Nh -u a415dzggdnkro3n:00ocfqin8hlcorr",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.split('{"access_token": "')[
        1].split('", "token_type":')[0]
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    if format == 'tex':
        # Handle .tex files by directly uploading their content
        dbx.files_upload(content.encode(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)
    else:
        dbx.files_upload(bs.getvalue(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)


def plot_forecast(df, var, label, file, trainMethod, sDepVar, transform='sum'):
    # First plot: Out of Sample
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df[df[trainMethod] == 0]['date'],
            df[df[trainMethod] == 0].groupby('date')[sDepVar].transform(transform), label='Actual',
            linestyle='dashed')
    ax.plot(df[df[trainMethod] == 0]['date'],
            df[df[trainMethod] == 0].groupby('date')[var].transform(transform), label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Contribution')
    ax.set_title('Out of Sample')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.5)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/{file}.png")
    plt.savefig(f"./Results/Presentation/{file}.svg")
    # Fixed the syntax error in upload call
    upload(plt, 'Project-based Internship', f'figures/{file}.png')

    # Second plot: Full Sample
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(df['date'],
            df.groupby('date')[sDepVar].transform(transform), label='Actual',
            linestyle='dashed')
    ax.plot(df['date'],
            df.groupby('date')[var].transform(transform), label=label)
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Contribution')
    ax.set_title('Full Sample')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.5)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/{file}_1.png")
    plt.savefig(f"./Results/Presentation/{file}_1.svg")
    # Fixed the syntax error in upload call
    upload(plt, 'Project-based Internship', f'figures/{file}_1.png')

    # Close all figures
    plt.close('all')

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

lDST = ['kbyg11', 'kbyg22', 'kbyg33_no_limitations', 'kbyg44_confidence_indicator']

# Write lDST to .AUX/
with open('./.AUX/lDST.txt', 'w') as lVars:
    lVars.write('\n'.join(lDST))

model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lDST])
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_0_dst.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

upload(ols, 'Project-based Internship', 'tables/3_0_dst.tex')

dfData['predicted_ols'] = y_scaler.inverse_transform(results.predict(dfDataScaled[lDST]).values.reshape(-1, 1))

# Predict and rescale sDepVar using OLS
dfData['predicted_dst'] = y_scaler.inverse_transform(results.predict(dfDataScaled[lDST]).values.reshape(-1, 1))

plot_forecast(dfData, 'predicted_dst', 'Predicted (DST)', '3_0_dst', trainMethod=trainMethod, sDepVar=sDepVar)

plot_forecast(dfData, 'predicted_dst', 'Predicted (DST)', '3_0_dst_mean', trainMethod=trainMethod, sDepVar=sDepVar,
         transform='mean')


# Calculate out-of-sample RMSE of DST
rmse_dst = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_dst'].replace(np.nan, 0)
                       ))
# symmetric Mean Absolute Error (sMAPE)
smape_dst = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_dst'])

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_dst'] = y_scaler.inverse_transform(results.predict(dfDataWIP[lDST]).values.reshape(-1, 1))

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
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_1_ols.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict and rescale sDepVar using OLS
dfData['predicted_ols'] = results.predict(dfDataScaled[lIndepVar])

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_1_ols.png")
plt.savefig("./Results/Presentation/3_1_ols.svg")

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_1_1_ols.png")
plt.savefig("./Results/Presentation/3_1_1_ols.svg")
plt.close('all')

# Calculate out-of-sample RMSE of OLS
rmse_ols = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_ols'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                  dfData[dfData[trainMethod] == 0]['predicted_ols'])

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_ols'] = y_scaler.inverse_transform(results.predict(dfDataWIP[lIndepVar]).values.reshape(-1, 1))

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

# Run OLS with lagged variables
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar_lag], missing='drop')
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_2_ols_lag.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict and rescale sDepVar using OLS with lagged variables
dfData['predicted_lag'] = results.predict(dfDataScaled[lIndepVar_lag])

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_3_ols_lag.png")
plt.savefig("./Results/Presentation/3_3_ols_lag.svg")

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_3_1_ols_lag.png")
plt.savefig("./Results/Presentation/3_3_1_ols_lag.svg")
plt.close('all')

# Calculate RMSE of OLS with lagged variables
rmse_ols_lag = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_lag'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                      dfData[dfData[trainMethod] == 0]['predicted_lag'])

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_lag'] = y_scaler.inverse_transform(results.predict(dfDataWIP[lIndepVar_lag]).values.reshape(-1, 1))

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

# Run OLS with lagged variables and budget
model = sm.OLS(dfDataScaledTrain[sDepVar], dfDataScaledTrain[lIndepVar_lag_budget], missing='drop')
results = model.fit()
# Save results to LaTeX
ols = results.summary(alpha=0.05).as_latex()

with open('Results/Tables/3_3_ols_lag_budget.tex', 'w', encoding='utf-8') as f:
    f.write(ols)

# Predict and rescale sDepVar using OLS with lagged variables and budget
dfData['predicted_lag_budget'] = results.predict(dfDataScaled[lIndepVar_lag_budget])

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_4_ols_lag_budget.png")
plt.savefig("./Results/Presentation/3_4_ols_lag_budget.svg")

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_4_1_ols_lag_budget.png")
plt.savefig("./Results/Presentation/3_4_1_ols_lag_budget.svg")
plt.close('all')

# Calculate RMSE of OLS with lagged variables and budget
rmse_ols_lag_budget = np.sqrt(mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                                                 dfData[dfData[trainMethod] == 0]['predicted_lag_budget'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_ols_lag_budget = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                             dfData[dfData[trainMethod] == 0]['predicted_lag_budget'])

# Predict dfDataWIP[sDepVar]
dfDataWIP['predicted_lag_budget'] = results.predict(dfDataWIP[lIndepVar_lag_budget])

dfDataWIP['predicted_lag_budget'] = y_scaler.inverse_transform(
    dfDataWIP['predicted_lag_budget'].values.reshape(-1, 1))

### Forecast Combination ###
# Produce a combined forecast of ols_lag_budget and pls
dfData['predicted_fc'] = (dfData['predicted_dst'] + dfData['predicted_lag_budget']) / 2

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_5_fc.png")
plt.savefig("./Results/Presentation/3_5_fc.svg")

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_5_1_fc.png")
plt.savefig("./Results/Presentation/3_5_1_fc.svg")
plt.close('all')

# Calculate RMSE of Forecast Combination
rmse_fc = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_fc'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_fc'])

# Compare RMSE and sMAPE of the different models in a table
dfRMSE = pd.DataFrame({'RMSE': [rmse_dst, rmse_ols, rmse_ols_lag, rmse_ols_lag_budget, rmse_fc],
                       'sMAPE': [smape_dst, smape_ols, smape_ols_lag, smape_ols_lag_budget, smape_fc]},
                      index=['DST', 'OLS', 'OLS with lagged variables', 'OLS with lagged variables and budget', 'FC'])

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

### Use clustering to find similar jobs and predict sDepVar for each cluster ###
lCluster = [2, 4, 6, 8, 10, 12, 14]
# For each cluster in cluster_{lCluster} do
for iCluster in lCluster:
    # Get the cluster labels to list using value_counts()
    lClusterLabels = dfData['cluster_' + str(iCluster)].value_counts().index.tolist()

    # For each cluster label in lClusterLabels do
    for iClusterLabel in lClusterLabels:
        # Run OLS
        model = sm.OLS(dfDataScaledTrain[dfDataScaledTrain['cluster_' + str(iCluster)] == iClusterLabel][sDepVar],
                       dfDataScaledTrain[dfDataScaledTrain['cluster_' + str(iCluster)] == iClusterLabel][
                           lIndepVar_lag_budget]
                       , missing='drop')
        results = model.fit()
        # Predict and rescale sDepVar using OLS with lagged variables and budget and add to cluster_{iCluster}
        dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
            iCluster)] = results.predict(
            dfDataScaled[dfDataScaled['cluster_' + str(iCluster)] == iClusterLabel][lIndepVar_lag_budget])
        dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
            iCluster)] = y_scaler.inverse_transform(
            dfData.loc[dfData['cluster_' + str(iCluster)] == iClusterLabel, 'predicted_cluster_' + str(
                iCluster)].values.reshape(-1, 1))

# Plot the sum of all predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_2'].transform('sum'),
        label='Predicted (2 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_4'].transform('sum'),
        label='Predicted (4 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_6'].transform('sum'),
        label='Predicted (6 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_8'].transform('sum'),
        label='Predicted (8 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_10'].transform('sum'),
        label='Predicted (10 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_12'].transform('sum'),
        label='Predicted (12 clusters)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_14'].transform('sum'),
        label='Predicted (14 clusters)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_6_cluster.png")
plt.savefig("./Results/Presentation/3_6_cluster.svg")
plt.close('all')

# Use Forecast Combination to combine the predictions of each cluster
# For each cluster in cluster_{lCluster} do
dfData['predicted_cluster_fc'] = (dfData['predicted_cluster_' + str(lCluster[0])]
                                  + dfData['predicted_cluster_' + str(lCluster[1])]
                                  + dfData['predicted_cluster_' + str(lCluster[2])]
                                  + dfData['predicted_cluster_' + str(lCluster[3])]
                                  + dfData['predicted_cluster_' + str(lCluster[4])]
                                  + dfData['predicted_cluster_' + str(lCluster[5])]
                                  + dfData['predicted_cluster_' + str(lCluster[6])]) / 7

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_7_fc_cluster.png")
plt.savefig("./Results/Presentation/3_7_fc_cluster.svg")

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'], dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'], dfData.groupby('date')['predicted_cluster_fc'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_7_1_fc_cluster.png")
plt.savefig("./Results/Presentation/3_7_1_fc_cluster.svg")
plt.close('all')

# Calculate RMSE of Forecast Combination
rmse_fc_cluster = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_cluster_fc'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc_cluster = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                         dfData[dfData[trainMethod] == 0]['predicted_cluster_fc'])

# Add RMSE and sMAPE of Forecast Combination to dfRMSE
dfRMSE.loc['FC_cluster'] = [rmse_fc_cluster, smape_fc_cluster]

### Combine Cluster Forecast Combination and DST ###

dfData['predicted_fc_cluster_dst'] = (dfData['predicted_cluster_fc'] + dfData['predicted_dst']) / 2

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
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_8_fc_cluster_dst.png")
plt.savefig("./Results/Presentation/3_8_fc_cluster_dst.svg")

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'], dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'], dfData.groupby('date')['predicted_fc_cluster_dst'].transform('sum'),
        label='Predicted (Forecast Combination)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/3_8_1_fc_cluster_dst.png")
plt.savefig("./Results/Presentation/3_8_1_fc_cluster_dst.svg")
plt.close('all')

# Calculate RMSE of Forecast Combination
rmse_fc_cluster_dst = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_fc_cluster_dst'].replace(np.nan, 0)))
# symmetric Mean Absolute Error (sMAPE)
smape_fc_cluster_dst = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                             dfData[dfData[trainMethod] == 0]['predicted_fc_cluster_dst'])

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
