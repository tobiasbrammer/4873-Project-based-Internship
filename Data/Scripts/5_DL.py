# Import required libraries
import os
import warnings
import runpy
import numpy as np
import pandas as pd
# import mlforecast
# import neuralforecast
# import keras
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
dfRMSE = pd.read_csv("./Results/Tables/3_4_rmse.csv", index_col=0)

# Rescale dfDataScaled to dfData
dfDataRescaled = dfDataScaled.copy()
dfDataRescaled[colIndepVarNum] = x_scaler.inverse_transform(dfDataScaled[colIndepVarNum].values)
dfDataRescaled[sDepVar] = y_scaler.inverse_transform(dfDataScaled[sDepVar].values.reshape(-1, 1))

train_index = dfData[dfData[trainMethod] == 0].index
dfDataScaledTrain = dfDataScaled.loc[train_index]
dfDataScaledTest = dfDataScaled.drop(train_index)

### LSTM ###
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

# Create model
iUnit = 256*4
model = Sequential()
model.add(LSTM(units=iUnit, return_sequences=True, input_shape=(
    dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=int(iUnit / 2), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=int(iUnit / 4), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=int(iUnit / 8), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=int(iUnit / 16), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=int(iUnit / 32), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=int(iUnit / 64), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='linear'))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)

# Fit model
start_time_lstm = datetime.datetime.now()
# Fit model to training data dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])]
model.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
          dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
          epochs=10,
          batch_size=512,
          validation_split=0.1,
          callbacks=[early_stop],
          verbose=0)
model.save('./.AUX/LSTM.h5')

# Predict and rescale using LSTM
dfData['predicted_lstm'] = pd.DataFrame(model.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])])[:, -1, 0]).values.reshape(-1, 1)

dfData['predicted_lstm'] = y_scaler.inverse_transform(dfData['predicted_lstm'].values.reshape(-1, 1))

end_time_lstm = datetime.datetime.now()
print(f'LSTM fit finished in {end_time_lstm - start_time_lstm}.')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_lstm'].transform('sum'),
        label='Predicted (LSTM)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/5_1_lstm.png")
plt.savefig("./Results/Presentation/5_1_lstm.svg")

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lstm'].transform('sum'),
        label='Predicted (LSTM)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/5_1_1_lstm.png")
plt.savefig("./Results/Presentation/5_1_1_lstm.svg")


# Calculate RMSE of LSTM
rmse_lstm = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_lstm']))
# Calculate sMAPE
smape_lstm = np.mean(np.abs(dfData[dfData[trainMethod] == 0]['predicted_lstm'] - dfData[dfData[trainMethod] == 0][sDepVar]) /
                     (np.abs(dfData[dfData[trainMethod] == 0][sDepVar]) + np.abs(
                         dfData[dfData[trainMethod] == 0]['predicted_lstm']))) * 100

# Add to dfRMSE
dfRMSE.loc['LSTM', 'RMSE'] = rmse_lstm
dfRMSE.loc['LSTM', 'sMAPE'] = smape_lstm

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

########################################################################################################################

# dfRMSE to latex
dfRMSE_latex = dfRMSE.copy()
# Bold the lowest RMSE
dfRMSE_latex.loc[dfRMSE['RMSE'] == dfRMSE_latex['RMSE'].min(), 'RMSE'] = '\\textbf{' + dfRMSE_latex['RMSE'].astype(
    str) + '}'
# Bold the lowest sMAPE
dfRMSE_latex.loc[dfRMSE['sMAPE'] == dfRMSE_latex['sMAPE'].min(), 'sMAPE'] = '\\textbf{' + dfRMSE_latex['sMAPE'].astype(
    str) + '}'

print(dfRMSE_latex)
