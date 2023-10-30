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
from sklearn.metrics import mean_squared_error
import multiprocessing

warnings.filterwarnings('ignore')

# Load ./dfData.parquet
if os.name == 'posix':
    sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
# If operating system is Windows then
elif os.name == 'nt':
    sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)


# Read token from Data/.AUX/dropbox.txt
with open('./.AUX/dropbox.txt', 'r') as f:
    token = f.read()

# Format as single line
token = token.replace('\n', '')

os.environ['DROPBOX'] = token


import dropbox
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt


def upload(ax, project, path):
    bs = BytesIO()
    format = path.split('.')[-1]
    ax.savefig(bs, bbox_inches='tight', format=format)

    token = os.getenv('DROPBOX')
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=bs.getvalue(),
        path=f'/Apps/Overleaf/{project}/{path}',
        mode=dropbox.files.WriteMode.overwrite)


# Load data
dfDataScaled = pd.read_parquet("./dfData_reg_scaled.parquet")
dfData = pd.read_parquet("./dfData_reg.parquet")
dfDataPred = pd.read_parquet("./dfDataPred.parquet")


# Define sMAPE
def smape(actual, predicted):
    return 100 / len(actual) * np.sum(np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted)))


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

train_index = dfData[dfData[trainMethod] == 1].index
dfDataScaledTrain = dfDataScaled.loc[train_index]
dfDataScaledTest = dfDataScaled.drop(train_index)

### LSTM ###
## Tune Hyperparameters using keras-tuner ##
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import keras_tuner as kt


def model_builder(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units_1', min_value=256, max_value=1028, step=8), return_sequences=True, input_shape=(
        dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].shape[1], 1)))
    model.add(Dropout(hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.05)))
    model.add(LSTM(units=hp.Int('units_2', min_value=0, max_value=512, step=8), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.05)))
    model.add(LSTM(units=hp.Int('units_3', min_value=0, max_value=256, step=8), return_sequences=False))
    model.add(Dropout(hp.Float('dropout_3', min_value=0.0, max_value=0.5, step=0.05)))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    return model

# Define tuner
tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=5,
                     factor=3,
                     directory='./.AUX',
                     project_name='LSTM')

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=5)

# Fit model
start_time_lstm_tune = datetime.datetime.now()
# Fit model to training data
tuner.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
             dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
             batch_size=int(
                 dfDataScaledTrain[lNumericCols][
                     dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].shape[
                     0] / 200),
             validation_split=0.25,
             callbacks=[early_stop],
             use_multiprocessing=True,
             workers=multiprocessing.cpu_count(),
             verbose=1)

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the LSTM
layer is {best_hps.get('units')}, and the optimal dropout rate is {best_hps.get('dropout')}.
""")


## Create model from optimal hyperparameters ##
model = tuner.hypermodel.build(best_hps)
# Fit model
model.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
          dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
          epochs=10,
          batch_size=int(
              dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].shape[
                  0] / 200),
          validation_split=0.25,
          callbacks=[early_stop],
          use_multiprocessing=True,
          workers=multiprocessing.cpu_count(),
          verbose=1)
model.save('./.AUX/LSTM_tune.tf')

# Plot loss
pd.DataFrame(model.history.history).plot(figsize=(20, 10))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of LSTM")
plt.grid(alpha=0.35)
plt.show()
plt.savefig("./Results/Figures/5_0_loss.png")
plt.savefig("./Results/Presentation/5_0_loss.svg")
upload(plt, 'Project-based Internship', 'figures/5_0_loss.png')

# Predict and rescale using LSTM
dfData['predicted_lstm'] = pd.DataFrame(
    model.predict(dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])]).reshape(-1, 1)
)

dfData['predicted_lstm'] = y_scaler.inverse_transform(dfData['predicted_lstm'].shift(-1).values.reshape(-1, 1))

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
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/5_1_lstm.png")
plt.savefig("./Results/Presentation/5_1_lstm.svg")
upload(plt, 'Project-based Internship', 'figures/5_1_lstm.png')

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
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/5_1_1_lstm.png")
plt.savefig("./Results/Presentation/5_1_1_lstm.svg")
upload(plt, 'Project-based Internship', 'figures/5_1_1_lstm.png')

# Calculate RMSE of LSTM
rmse_lstm = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_lstm']))
# Calculate sMAPE
smape_lstm = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_lstm'])

# Add to dfRMSE
dfRMSE.loc['LSTM', 'RMSE'] = rmse_lstm
dfRMSE.loc['LSTM', 'sMAPE'] = smape_lstm

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)


########################################################################################################################

# Calculate average of all columns in dfDataPred except 'date', 'job_no' and sDepVar
dfDataPred['predicted_avg'] = dfDataPred[dfDataPred.columns.difference(['date', 'job_no', sDepVar])].mean(axis=1)

########################################################################################################################

# dfRMSE to latex
dfRMSE_latex = dfRMSE.copy()
dfRMSE_latex = dfRMSE_latex.round(4)
dfRMSE_latex['RMSE'] = dfRMSE_latex['RMSE'].apply(lambda x: '{0:.4f}'.format(x))
dfRMSE_latex['sMAPE'] = dfRMSE_latex['sMAPE'].apply(lambda x: '{0:.4f}'.format(x))

# Bold the lowest RMSE to save to .tex
dfRMSE_latex.loc[dfRMSE_latex['RMSE'] == dfRMSE_latex['RMSE'].min(), 'RMSE'] = r'\textbf{' + dfRMSE_latex.loc[
    dfRMSE_latex['RMSE'] == dfRMSE_latex['RMSE'].min(), 'RMSE'].astype(str) + '}'
# Bold the lowest sMAPE to save to .tex
dfRMSE_latex.loc[dfRMSE_latex['sMAPE'] == dfRMSE_latex['sMAPE'].min(), 'sMAPE'] = r'\textbf{' + dfRMSE_latex.loc[
    dfRMSE_latex['sMAPE'] == dfRMSE_latex['sMAPE'].min(), 'sMAPE'].astype(str) + '}'

print(dfRMSE_latex)

# Save dfRMSE to .tex
with open('./Results/Tables/5_1_rmse.tex', 'w') as f:
    f.write(dfRMSE_latex.to_latex())

# Save to .parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")

########################################################################################################################
# if ./Results/Figures/Jobs does not exist, create it

# if not os.path.exists('./Results/Figures/Jobs'):
#     os.makedirs('./Results/Figures/Jobs')
#
# ## For each job_no plot the actual and predicted sDepVar
# for job_no in dfDataPred['job_no'].unique():
#     # Get the data of job_no
#     dfDataJob = dfDataPred[dfDataPred['job_no'] == job_no]
#     # Plot the actual and predicted contribution of sJobNo
#     fig, ax = plt.subplots(figsize=(20, 10))
#     for col in dfDataJob.columns:
#         if col not in ['date', 'job_no']:
#             ax.plot(dfDataJob['date'], dfDataJob[col], label=col)
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Contribution')
#     ax.set_title(f'Actual vs. Predicted Contribution of {job_no}')
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
#
#     plt.grid(alpha=0.5)
#     plt.rcParams['axes.axisbelow'] = True
#     plt.savefig(f"./Results/Figures/Jobs/{job_no}.png")
#
#     # Plot the cumsum of actual and predicted contribution of sJobNo
#     fig, ax = plt.subplots(figsize=(20, 10))
#     for col in dfDataJob.columns:
#         if col not in ['date', 'job_no']:
#             ax.plot(dfDataJob['date'], dfDataJob[col].cumsum(), label=col)
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Cumulative Contribution')
#     ax.set_title(f'Actual vs. Predicted Cumulative Contribution of {job_no}')
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
#
#     plt.grid(alpha=0.5)
#     plt.rcParams['axes.axisbelow'] = True
#     plt.savefig(f"./Results/Figures/Jobs/{job_no}_sum.png")

########################################################################################################################
