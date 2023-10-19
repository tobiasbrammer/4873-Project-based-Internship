# Import required libraries
import os
import warnings
import runpy
import numpy as np
import pandas as pd
import datetime
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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings('ignore')

# Load ./dfData.parquet
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
# sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)

# Load data
dfDataScaled = pd.read_parquet("./dfData_reg_scaled.parquet")
dfData = pd.read_parquet("./dfData_reg.parquet")

# Import lNumericCols from ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'r') as f:
    lNumericCols = f.read()
lNumericCols = lNumericCols.split('\n')

# Replace infinite values with NaN
dfDataScaled.replace([np.inf, -np.inf], np.nan, inplace=True)
dfData.replace([np.inf, -np.inf], np.nan, inplace=True)

# Keep only numeric columns
dfDataScaled = dfDataScaled[lNumericCols + ['train']]
# dfData = dfData[lNumericCols + ['train']]

# Replace NaN with 0
dfDataScaled.fillna(0, inplace=True)
dfData[lNumericCols].fillna(0, inplace=True)

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

train_index = dfData[dfData['train'] == 1].index
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
param_grid = {
    'alpha': np.arange(0.1, 10, 0.1),
    'l1_ratio': np.arange(0.001, 1.00, 0.001),
    'tol': [0.0001, 0.001]
}
# Define randomized search
elastic_net = ElasticNet(tol=1e-4, random_state=0)
elastic_net_cv_sparse = RandomizedSearchCV(elastic_net, param_grid, n_iter=1000, scoring=None, cv=3, verbose=0,
                                           refit=True)
elastic_net_cv_full = RandomizedSearchCV(elastic_net, param_grid, n_iter=1000, scoring=None, cv=3, verbose=0,
                                         refit=True)

# Fit to the training data

# Sparse model with OLS variables
start_time_en_sparse = datetime.datetime.now()
elastic_net_cv_sparse.fit(dfDataScaledTrain[lIndepVar_lag_budget], dfDataScaledTrain[sDepVar])
dfData['predicted_en_sparse'] = elastic_net_cv_sparse.predict(dfDataScaled[lIndepVar_lag_budget])
dfData['predicted_en_sparse'] = y_scaler.inverse_transform(dfData['predicted_en_sparse'].values.reshape(-1, 1))
end_time_en_sparse = datetime.datetime.now()
print(f'Sparse EN fit finished in {end_time_en_sparse - start_time_en_sparse}.')

# All variables
start_time_en_full = datetime.datetime.now()
elastic_net_cv_full.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                        dfDataScaledTrain[sDepVar])
dfData['predicted_en_full'] = elastic_net_cv_full.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])])
dfData['predicted_en_full'] = y_scaler.inverse_transform(dfData['predicted_en_full'].values.reshape(-1, 1))
end_time_en_full = datetime.datetime.now()
print(f'Full EN fit finished in {end_time_en_full - start_time_en_full}.')

# Group by date and sum over all jobs
dfData['sum_predicted_en_sparse'] = dfData.groupby('date')['predicted_en_sparse'].transform('sum')
dfData['sum_predicted_en_full'] = dfData.groupby('date')['predicted_en_full'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_en_sparse'], label='Predicted (Elastic Net, Sparse Variables)')
ax.plot(dfData['date'], dfData['sum_predicted_en_full'], label='Predicted (Elastic Net, All Variables)')
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
plt.savefig("./Results/Figures/4_0_en.png")
plt.savefig("./Results/Presentation/4_0_en.svg")
plt.show()

# Calculate RMSE of EN
rmse_en_sparse = np.sqrt(
    mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_en_sparse']))
rmse_en_full = np.sqrt(
    mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_en_full']))
# Calculate sMAPE
smape_en_sparse = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_en_sparse']) /
                   (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                       dfData[dfData['train'] == 0]['predicted_en_sparse']))) * 100
smape_en_full = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_en_full']) /
                      (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                            dfData[dfData['train'] == 0]['predicted_en_full']))) * 100


# Add to dfRMSE
dfRMSE.loc['Elastic Net (Sparse)', 'RMSE'] = rmse_en_sparse
dfRMSE.loc['Elastic Net (Sparse)', 'sMAPE'] = smape_en_sparse
dfRMSE.loc['Elastic Net (Full)', 'RMSE'] = rmse_en_full
dfRMSE.loc['Elastic Net (Full)', 'sMAPE'] = smape_en_full


### Random Forest Regression ###
# Define Random Forest model
rf = RandomForestRegressor(random_state=0)

## Define hyperparameter grid ##
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=1, stop=100, num=21)]
# Number of features to consider at every split
# max_features = ['auto', 'sqrt']
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 100, num=50)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Define randomized search
rf_cv = RandomizedSearchCV(rf, random_grid, n_iter=1000, scoring=None, cv=3, verbose=0, refit=True)

# Fit to the training data
start_time_rf = datetime.datetime.now()
rf_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
          dfDataScaledTrain[sDepVar])
# Predict and rescale using RF
dfData['predicted_rf'] = rf_cv.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])])
dfData['predicted_rf'] = y_scaler.inverse_transform(dfData['predicted_rf'].values.reshape(-1, 1))
end_time_rf = datetime.datetime.now()
print(f'RF fit finished in {end_time_rf - start_time_rf}.')

# Group by date and sum over all jobs
dfData['sum_predicted_rf'] = dfData.groupby('date')['predicted_rf'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_rf'], label='Predicted (Random Forest)')
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
plt.savefig("./Results/Figures/4_1_rf.png")
plt.savefig("./Results/Presentation/4_1_rf.svg")
plt.show()

# Calculate RMSE of RF
rmse_rf = np.sqrt(mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_rf']))
# Calculate sMAPE
smape_rf = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_rf']) /
                     (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                            dfData[dfData['train'] == 0]['predicted_rf']))) * 100

# Add to dfRMSE
dfRMSE.loc['Random Forest', 'RMSE'] = rmse_rf
dfRMSE.loc['Random Forest', 'sMAPE'] = smape_rf

### LSTM ###
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

# Create model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(dfDataScaledTrain.shape[1] - 1, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))

model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# Fit model
start_time_lstm = datetime.datetime.now()
model.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].values.reshape(
    dfDataScaledTrain.shape[0], dfDataScaledTrain.shape[1] - 1, 1),
            dfDataScaledTrain[sDepVar].values.reshape(dfDataScaledTrain.shape[0], 1),
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop])
end_time_lstm = datetime.datetime.now()
print(f'LSTM fit finished in {end_time_lstm - start_time_lstm}.')

# Predict and rescale using LSTM
dfData['predicted_lstm'] = model.predict(dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].values.reshape(
    dfDataScaled.shape[0], dfDataScaled.shape[1] - 1, 1))
dfData['predicted_lstm'] = y_scaler.inverse_transform(dfData['predicted_lstm'].values.reshape(-1, 1))

# Group by date and sum over all jobs
dfData['sum_predicted_lstm'] = dfData.groupby('date')['predicted_lstm'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_lstm'], label='Predicted (LSTM)')
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
plt.savefig("./Results/Figures/4_2_lstm.png")
plt.savefig("./Results/Presentation/4_2_lstm.svg")
plt.show()

# Calculate RMSE of LSTM
rmse_lstm = np.sqrt(mean_squared_error(dfData[dfData['train'] == 0][sDepVar], dfData[dfData['train'] == 0]['predicted_lstm']))
# Calculate sMAPE
smape_lstm = np.mean(np.abs(dfData[dfData['train'] == 0][sDepVar] - dfData[dfData['train'] == 0]['predicted_lstm']) /
                        (np.abs(dfData[dfData['train'] == 0][sDepVar]) + np.abs(
                            dfData[dfData['train'] == 0]['predicted_lstm']))) * 100

print(dfRMSE)

# dfRMSE to latex
dfRMSE_latex = dfRMSE.copy()
# Bold the lowest RMSE
dfRMSE_latex.loc[dfRMSE['RMSE'] == dfRMSE_latex['RMSE'].min(), 'RMSE'] = '\\textbf{' + dfRMSE_latex['RMSE'].astype(str) + '}'
# Bold the lowest sMAPE
dfRMSE_latex.loc[dfRMSE['sMAPE'] == dfRMSE_latex['sMAPE'].min(), 'sMAPE'] = '\\textbf{' + dfRMSE_latex['sMAPE'].astype(str) + '}'

print(dfRMSE_latex)