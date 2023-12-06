# Import required libraries
import sys
sys.path.insert(0, '.')
import os
import warnings
import numpy as np
import pandas as pd
import datetime
import joblib
from plot_config import *
from plot_predicted import *
from predict_and_scale import *
from notify import *
from smape import *
from sklearn.metrics import mean_squared_error
import multiprocessing
import keras_core as keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
import keras_tuner as kt
from keras.callbacks import Callback


class TerminateOnThreshold(Callback):
    def __init__(self, threshold):
        super(TerminateOnThreshold, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        if loss is not None and loss > self.threshold:
            self.model.stop_training = True
            print(f"\nEpoch {epoch}: Loss exceeded the threshold of {self.threshold}. Training stopped.")


warnings.filterwarnings('ignore')


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
dfDataPred = pd.read_parquet("./dfDataPred.parquet")
dfDataWIP = pd.read_parquet("./dfDataWIP_pred.parquet")

# Import lNumericCols from ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'r') as f:
    lNumericCols = f.read()
lNumericCols = lNumericCols.split('\n')

# Replace infinite values with NaN
dfDataScaled.replace([np.inf, -np.inf], np.nan, inplace=True)
dfData.replace([np.inf, -np.inf], np.nan, inplace=True)

# Keep only numeric columns
dfDataScaled = dfDataScaled[lNumericCols + ['train']]

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

# Import lIndepVar
with open('./.AUX/lIndepVar.txt', 'r') as f:
    lIndepVar = f.read()

# Import sDepVar from ./.AUX/sDepVar.txt
with open('./.AUX/sDepVar.txt', 'r') as f:
    sDepVar = f.read()

# Load trainMethod from ./.AUX/trainMethod.txt
with open('./.AUX/trainMethod.txt', 'r') as f:
    trainMethod = f.read()

# Import lJobNo from ./.AUX/lJobNo.txt
with open('./.AUX/lJobNo.txt', 'r') as f:
    lJobNo = f.read()
lJobNo = lJobNo.split('\n')

# Import lJobNoWIP from ./.AUX/lJobNoWIP.txt
with open('./.AUX/lJobNoWIP.txt', 'r') as f:
    lJobNoWIP = f.read()
lJobNoWIP = lJobNoWIP.split('\n')

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


def model_builder(hp):
    # Use sMAPE as loss function
    import keras.backend as k
    def rmse_loss(y_true, y_pred):
        return k.sqrt(k.mean(k.square(y_true - y_pred)))

    def smape_loss(y_true, y_pred):
        return 100 * k.mean(k.abs(y_true - y_pred) / (k.abs(y_pred) + k.abs(y_true)), axis=-1)


    model = Sequential()
    # First LSTM layer
    model.add(LSTM(units=hp.Choice('input_unit_init', values=[2 ** n for n in range(3, 8)]),
                   return_sequences=True,
                   input_shape=(
                       dfDataScaledTrain[lNumericCols][
                           dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].shape[
                           1], 1)))
    # Subsequent LSTM layers
    for l in range(hp.Int('additional_layers', 0, 4)):
        # Check if l == 0
        if l == 0:  # Add nothing
            pass
        # Check if it's the last LSTM layer
        if l == hp.Int('additional_layers', 1, 4) - 1:
            # Last LSTM layer
            model.add(
                LSTM(units=hp.Choice(f'input_unit_{l + 1}', values=[2 ** n for n in range(2, 7)]),
                     return_sequences=False))
        else:
            # Not the last LSTM layer
            model.add(
                LSTM(units=hp.Choice(f'input_unit_{l + 1}', values=[2 ** n for n in range(2, 7)]),
                     return_sequences=True))
        # Add dropout
        model.add(Dropout(hp.Float(f'dropout_{l + 1}', min_value=0.0, max_value=0.5, step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',
                                            values=['relu', 'sigmoid', 'linear', 'tanh', 'exponential'],
                                            default='sigmoid')))
    # Try with different optimizers
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'adagrad'])
    # Compile model
    model.compile(optimizer=optimizer, loss=smape_loss, metrics=[rmse_loss])
    return model


# Define tuner
tuner_2 = kt.Hyperband(model_builder,
                       objective='val_loss',
                       max_epochs=50,
                       factor=3,
                       seed=607,
                       directory='./.MODS',
                       project_name='LSTM_2')

tuner_4 = kt.Hyperband(model_builder,
                       objective='val_loss',
                       max_epochs=50,
                       factor=3,
                       seed=607,
                       directory='./.MODS',
                       project_name='LSTM_4')

tuner_8 = kt.Hyperband(model_builder,
                       objective='val_loss',
                       max_epochs=50,
                       factor=3,
                       seed=607,
                       directory='./.MODS',
                       project_name='LSTM_8')

# Define tuner
tuner_16 = kt.Hyperband(model_builder,
                        objective='val_loss',
                        max_epochs=50,
                        factor=3,
                        seed=607,
                        directory='./.MODS',
                        project_name='LSTM_16')

# Define tuner
tuner_32 = kt.Hyperband(model_builder,
                        objective='val_loss',
                        max_epochs=50,
                        factor=3,
                        seed=607,
                        directory='./.MODS',
                        project_name='LSTM_32')

# Define tuner
tuner_64 = kt.Hyperband(model_builder,
                        objective='val_loss',
                        max_epochs=50,
                        max_retries_per_trial=5,
                        factor=3,
                        seed=607,
                        directory='./.MODS',
                        project_name='LSTM_64')

tuner_128 = kt.Hyperband(model_builder,
                         objective='val_loss',
                         max_epochs=50,
                         factor=3,
                         seed=607,
                         directory='./.MODS',
                         project_name='LSTM_128')

# Define early stopping
early_stop = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=15)

# Search for optimal hyperparameters
start_time_lstm_tune = datetime.datetime.now()

# Fit model to training data
tuner_2.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                batch_size=2,
                validation_split=0.10,
                callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

# Fit model to training data
tuner_4.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
               dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
               batch_size=4,
               validation_split=0.10,
               callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
               use_multiprocessing=True,
               workers=multiprocessing.cpu_count(),
               verbose=1)

tuner_8.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
               dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
               batch_size=8,
               validation_split=0.10,
               callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
               use_multiprocessing=True,
               workers=multiprocessing.cpu_count(),
               verbose=1)

tuner_16.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                batch_size=16,
                validation_split=0.10,
                callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

tuner_32.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                batch_size=32,
                validation_split=0.10,
                callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

tuner_64.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                batch_size=64,
                validation_split=0.10,
                callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

tuner_128.search(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                  dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                  batch_size=128,
                  validation_split=0.10,
                  callbacks=[early_stop, TerminateOnNaN(), TerminateOnThreshold(50)],
                  use_multiprocessing=True,
                  workers=multiprocessing.cpu_count(),
                  verbose=1)

# Get the optimal hyperparameters
best_hps_2 = tuner_2.get_best_hyperparameters()
best_hps_4 = tuner_4.get_best_hyperparameters()
best_hps_8 = tuner_8.get_best_hyperparameters()
best_hps_16 = tuner_16.get_best_hyperparameters()
best_hps_32 = tuner_32.get_best_hyperparameters()
best_hps_64 = tuner_64.get_best_hyperparameters()
best_hps_128 = tuner_128.get_best_hyperparameters()
# Fit using models
model_fit_2 = tuner_2.hypermodel.build(best_hps_2)
model_fit_4 = tuner_4.hypermodel.build(best_hps_4)
model_fit_8 = tuner_8.hypermodel.build(best_hps_8)
model_fit_16 = tuner_16.hypermodel.build(best_hps_16)
model_fit_32 = tuner_32.hypermodel.build(best_hps_32)
model_fit_64 = tuner_64.hypermodel.build(best_hps_64)
model_fit_128 = tuner_128.hypermodel.build(best_hps_128)

# Save models to use later without retraining
model_fit_2.save('./.MODS/LSTM_2.tf')
model_fit_4.save('./.MODS/LSTM_4.tf')
model_fit_8.save('./.MODS/LSTM_8.tf')
model_fit_16.save('./.MODS/LSTM_16.tf')
model_fit_32.save('./.MODS/LSTM_32.tf')
model_fit_64.save('./.MODS/LSTM_64.tf')
model_fit_128.save('./.MODS/LSTM_128.tf')


# Fit model
model_fit_2.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                epochs=250,
                batch_size=2,
                validation_split=0.1,
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

model_fit_4.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                epochs=250,
                batch_size=4,
                validation_split=0.1,
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

model_fit_8.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                epochs=250,
                batch_size=8,
                validation_split=0.1,
                use_multiprocessing=True,
                workers=multiprocessing.cpu_count(),
                verbose=1)

model_fit_16.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                 dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                 epochs=250,
                 batch_size=16,
                 validation_split=0.1,
                 use_multiprocessing=True,
                 workers=multiprocessing.cpu_count(),
                 verbose=1)

model_fit_32.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                 dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                 epochs=250,
                 batch_size=32,
                 validation_split=0.1,
                 use_multiprocessing=True,
                 workers=multiprocessing.cpu_count(),
                 verbose=1)

model_fit_64.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                 dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                 epochs=250,
                 batch_size=64,
                 validation_split=0.1,
                 use_multiprocessing=True,
                 workers=multiprocessing.cpu_count(),
                 verbose=1)

model_fit_128.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
                  dfDataScaledTrain[sDepVar].values.reshape(-1, 1),
                  epochs=250,
                  batch_size=128,
                  validation_split=0.1,
                  use_multiprocessing=True,
                  workers=multiprocessing.cpu_count(),
                  verbose=1)

# Get val_loss of best model
val_loss_2 = tuner_2.oracle.get_best_trials()[0].score
val_loss_4 = tuner_4.oracle.get_best_trials()[0].score
val_loss_8 = tuner_8.oracle.get_best_trials()[0].score
val_loss_16 = tuner_16.oracle.get_best_trials()[0].score
val_loss_32 = tuner_32.oracle.get_best_trials()[0].score
val_loss_64 = tuner_64.oracle.get_best_trials()[0].score
val_loss_128 = tuner_128.oracle.get_best_trials()[0].score

# Find the best model and set best_hps
# Make a dataframe of the val_loss of the best models
df_val_loss = pd.DataFrame({'batch_size': [2, 4, 8, 16, 32, 64, 128],
                            'val_loss': [val_loss_2, val_loss_4, val_loss_8, val_loss_16, val_loss_32, val_loss_64, val_loss_128]})
# Find the batch size with the lowest val_loss
best_batch_size = df_val_loss[df_val_loss['val_loss'] == df_val_loss['val_loss'].min()]['batch_size'].values[0]
# Set best_hps: best_hps = tuner_{best_batch_size}.get_best_hyperparameters()[0]
best_hps = eval(f'best_hps_{best_batch_size}')

iRollWindow = 10

# Compare val_loss of best models over epochs
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(pd.Series(model_fit_4.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='4 batch size',
        linestyle='solid' if best_batch_size == 4 else 'dashed')
ax.plot(pd.Series(model_fit_8.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='8 batch size',
        linestyle='solid' if best_batch_size == 8 else 'dashed')
ax.plot(pd.Series(model_fit_16.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='16 batch size',
        linestyle='solid' if best_batch_size == 16 else 'dashed')
ax.plot(pd.Series(model_fit_32.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='32 batch size',
        linestyle='solid' if best_batch_size == 32 else 'dashed')
ax.plot(pd.Series(model_fit_64.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='64 batch size',
        linestyle='solid' if best_batch_size == 64 else 'dashed')
ax.plot(pd.Series(model_fit_128.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='128 batch size',
        linestyle='solid' if best_batch_size == 128 else 'dashed')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.set_xlim(0, len(model_fit_4.history.history['val_loss']))
plt.title("Rolling Validation Loss of LSTM")
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/5_0_lstm_tune.png")
plt.savefig("./Results/Presentation/5_0_lstm_tune.svg")
upload(plt, 'Project-based Internship', 'figures/5_0_lstm_tune.png')

# Print optimal hyperparameters. Account for the fact that the number of layers is not the same as the number of units
print(f"The optimal batch size is {best_batch_size}.")
print(f"""The optimal number of units in the first LSTM layer is {best_hps.get('input_unit_init')}.
The optimal number of additional layers is {best_hps.get('additional_layers')}.""")
for i in range(best_hps.get('additional_layers')):
    print(f"""The optimal number of units in the {i + 1}. hidden layer is {best_hps.get(f'input_unit_{i + 1}')}.
    With an optimal dropout of  {round(best_hps.get(f'dropout_{i + 1}'), 2)}. """)
print(f"""The optimal activation function in the output layer is {best_hps.get('dense_activation')}.""")

## Create model from optimal hyperparameters ##
early_stop = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=50)

# Fit model
model_fit = model_builder(best_hps)

model_fit.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])],
              dfDataScaledTrain[sDepVar],
              epochs=500,
              batch_size=best_batch_size,
              validation_split=0.1,
              callbacks=[early_stop, TerminateOnNaN()],
              use_multiprocessing=True,
              workers=multiprocessing.cpu_count(),
              verbose=1)

model_fit.save('./.MODS/LSTM_tune.tf')
model_fit.summary()

# Plot loss
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(pd.Series(model_fit.history.history['loss']).rolling(iRollWindow).mean()[iRollWindow:], label='Training')
ax.plot(pd.Series(model_fit.history.history['val_loss']).rolling(iRollWindow).mean()[iRollWindow:], label='Validation')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss of LSTM")
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/5_0_loss.png")
plt.savefig("./Results/Presentation/5_0_loss.svg")
upload(plt, 'Project-based Internship', 'figures/5_0_loss.png')

predict_and_scale(dfData, dfDataScaled, model_fit, 'lstm',
                  dfDataScaled[lNumericCols].columns.difference([sDepVar]),
                  lJobNo,
                  bConst=False,
                  iBatchSize=best_batch_size)

predict_and_scale(dfData, dfDataScaled, model_fit_32, 'lstm_32',
                  dfDataScaled[lNumericCols].columns.difference([sDepVar]),
                  lJobNo,
                  bConst=False,
                  iBatchSize=32
                  )

predict_and_scale(dfData, dfDataScaled, model_fit_64, 'lstm_64',
                    dfDataScaled[lNumericCols].columns.difference([sDepVar]),
                    lJobNo,
                    bConst=False,
                    iBatchSize=64
                    )

predict_and_scale(dfData, dfDataScaled, model_fit_128, 'lstm_128',
                    dfDataScaled[lNumericCols].columns.difference([sDepVar]),
                    lJobNo,
                    bConst=False,
                    iBatchSize=128
                    )



# Plot difference between predicted and actual values
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'),
        label='Actual', linestyle='dashed')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lstm'].transform('sum'),
        label='LSTM (best)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lstm_32'].transform('sum'),
        label='LSTM (batch size = 32)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lstm_64'].transform('sum'),
        label='LSTM (batch size = 64)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lstm_128'].transform('sum'),
        label='LSTM (batch size = 128)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
plt.xlabel("Date")
plt.ylabel("Predicted")
plt.title("Predicted WIP")
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/5_1_lstm_batch.png")
plt.savefig("./Results/Presentation/5_1_lstm_batch.svg")
upload(plt, 'Project-based Internship', 'figures/5_1_lstm_batch.png')

plot_predicted(dfData, 'predicted_lstm', 'LSTM', '5_1_lstm', transformation='sum', trainMethod=trainMethod,
               sDepVar=sDepVar)

# Calculate RMSE of LSTM
rmse_lstm = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar],
                       dfData[dfData[trainMethod] == 0]['predicted_lstm'].replace(np.nan, 0)))
# Calculate sMAPE
smape_lstm = smape(dfData[dfData[trainMethod] == 0][sDepVar],
                   dfData[dfData[trainMethod] == 0]['predicted_lstm'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['LSTM', 'RMSE'] = rmse_lstm
dfRMSE.loc['LSTM', 'sMAPE'] = smape_lstm

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)

# Add to dfDataPred
dfDataPred['predicted_lstm'] = dfData['predicted_lstm']

# Predict WIP
predict_and_scale(dfDataWIP, dfDataWIP.replace(np.nan, 0), model_fit, 'lstm',
                  dfDataScaled[lNumericCols].columns.difference([sDepVar]),
                  lJobNoWIP,
                  bConst=False,
                  iBatchSize=best_batch_size)

print(f'LSTM finished in {datetime.datetime.now() - start_time_lstm_tune}.')

########################################################################################################################

# Calculate average of all columns in dfDataPred except 'date', 'job_no' and sDepVar
dfDataPred['predicted_avg'] = dfDataPred[['predicted_boost',
                                          'predicted_en',
                                          'predicted_gb',
                                          'predicted_lag',
                                          'predicted_lag_budget',
                                          'predicted_lstm',
                                          'predicted_ols',
                                          'predicted_rf_full',
                                          'predicted_rf_sparse',
                                          'predicted_et',
                                          'predicted_xgb']].mean(axis=1)
dfData['predicted_avg'] = dfDataPred['predicted_avg']

dfDataPred[sDepVar] = dfData[sDepVar]

dfDataWIP['predicted_avg'] = dfDataWIP[['predicted_boost',
                                        'predicted_en',
                                        'predicted_gb',
                                        'predicted_lag',
                                        'predicted_lag_budget',
                                        'predicted_lstm',
                                        'predicted_ols',
                                        'predicted_rf_full',
                                        'predicted_rf_sparse',
                                        'predicted_et',
                                        'predicted_xgb']].mean(axis=1)

### Explore different weighting schemes ###
# Calculate covariance matrix of the forecast errors. The forecast errors are the difference between the actual and
# predicted values of sDepVar.
dfDataPredError = pd.DataFrame()
for col in ['predicted_en', 'predicted_gb', 'predicted_lag', 'predicted_lag_budget',
            'predicted_lstm', 'predicted_ols', 'predicted_rf_full', 'predicted_rf_sparse',
            'predicted_et', 'predicted_xgb']:
    dfDataPredError[f'{col}'] = pd.DataFrame(np.abs(dfDataPred[col] - dfDataPred[sDepVar])).mean(axis=0)

# Calculate the weights as dfDataPredError.transpose() / sum(dfDataPredError.transpose())
dfDataPredWeights = pd.DataFrame()
dfDataPredWeights['weights'] = dfDataPredError.transpose() / dfDataPredError.sum(axis=1)[0]

# Calculate the weighted average of the predicted values
dfDataPred['predicted_bates_granger'] = dfDataPred[['predicted_boost',
                                                    'predicted_en',
                                                    'predicted_gb',
                                                    'predicted_lag',
                                                    'predicted_lag_budget',
                                                    'predicted_lstm',
                                                    'predicted_ols',
                                                    'predicted_rf_full',
                                                    'predicted_rf_sparse',
                                                    'predicted_et',
                                                    'predicted_xgb']].mul(dfDataPredWeights['weights'],
                                                                          axis=1).sum(axis=1)
dfData['predicted_bates_granger'] = dfDataPred['predicted_bates_granger']

dfDataWIP['predicted_bates_granger'] = dfDataWIP[['predicted_boost',
                                                  'predicted_en',
                                                  'predicted_gb',
                                                  'predicted_lag',
                                                  'predicted_lag_budget',
                                                  'predicted_lstm',
                                                  'predicted_ols',
                                                  'predicted_rf_full',
                                                  'predicted_rf_sparse',
                                                  'predicted_et',
                                                  'predicted_xgb']].mul(dfDataPredWeights['weights'],
                                                                        axis=1).sum(axis=1)

# Calculate RMSE of Bates and Granger
rmse_bates_granger = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_bates_granger'].replace(np.nan, 0)))
# Calculate sMAPE
smape_bates_granger = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                            dfData[dfData[trainMethod] == 0]['predicted_bates_granger'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Bates and Granger', 'RMSE'] = rmse_bates_granger
dfRMSE.loc['Bates and Granger', 'sMAPE'] = smape_bates_granger

## MSE-based weights as ω = MSE^(-1) / sum(MSE^(-1))
# Calculate MSE
dfDataPredMSE = pd.DataFrame()
for col in ['predicted_en', 'predicted_gb', 'predicted_lag', 'predicted_lag_budget',
            'predicted_lstm', 'predicted_ols', 'predicted_rf_full', 'predicted_rf_sparse',
            'predicted_et', 'predicted_xgb']:
    dfDataPredMSE[f'{col}'] = pd.DataFrame((dfDataPred[col] - dfDataPred[sDepVar]) ** 2).mean(axis=0)

# Calculate the weights as dfDataPredMSE.transpose() / sum(dfDataPredMSE.transpose())
dfDataPredWeightsMSE = pd.DataFrame()
dfDataPredWeightsMSE['weights'] = dfDataPredMSE.transpose() / dfDataPredMSE.sum(axis=1)[0]

# Calculate the weighted average of the predicted values
dfDataPred['predicted_mse'] = dfDataPred[['predicted_boost',
                                          'predicted_en',
                                          'predicted_gb',
                                          'predicted_lag',
                                          'predicted_lag_budget',
                                          'predicted_lstm',
                                          'predicted_ols',
                                          'predicted_rf_full',
                                          'predicted_rf_sparse',
                                          'predicted_et',
                                          'predicted_xgb']].mul(
    dfDataPredWeightsMSE['weights'], axis=1).sum(axis=1)

dfData['predicted_mse'] = dfDataPred['predicted_mse']
dfDataPred['contribution_cumsum'] = dfData['contribution_cumsum']

dfDataWIP['predicted_mse'] = dfDataWIP[['predicted_boost',
                                        'predicted_en',
                                        'predicted_gb',
                                        'predicted_lag',
                                        'predicted_lag_budget',
                                        'predicted_lstm',
                                        'predicted_ols',
                                        'predicted_rf_full',
                                        'predicted_rf_sparse',
                                        'predicted_et',
                                        'predicted_xgb']].mul(dfDataPredWeightsMSE['weights'],
                                                              axis=1).sum(axis=1)

# Calculate RMSE of MSE
rmse_mse = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_mse'].replace(np.nan, 0)))
# Calculate sMAPE
smape_mse = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                  dfData[dfData[trainMethod] == 0]['predicted_mse'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['MSE', 'RMSE'] = rmse_mse
dfRMSE.loc['MSE', 'sMAPE'] = smape_mse

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')[sDepVar].transform('sum'),
        label='Actual', linestyle='dashed')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_cluster_fc'].transform('sum'),
        label='Predicted (clustered ols)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_boost'].transform('sum'),
        label='Predicted (boosting)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_lstm'].transform('sum'),
        label='Predicted (lstm)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_rf_et'].transform('sum'),
        label='Predicted (ensemble)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_avg'].transform('sum'),
        label='Predicted (avg)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_bates_granger'].transform('sum'),
        label='Predicted (bates granger)')
ax.plot(dfData[dfData[trainMethod] == 0]['date'],
        dfData[dfData[trainMethod] == 0].groupby('date')['predicted_mse'].transform('sum'),
        label='Predicted (mse)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Out of Sample')
ax.set_aspect('auto')
ax.set_ylim([-5, 15.00])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/5_2_avg.png")
plt.savefig("./Results/Presentation/5_2_avg.svg")
upload(plt, 'Project-based Internship', 'figures/5_2_avg.png')

# Plot the sum of predicted and actual sDepVar by date (full sample)
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(dfData['date'],
        dfData.groupby('date')[sDepVar].transform('sum'), label='Actual', linestyle='dashed')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_cluster_fc'].transform('sum'),
        label='Predicted (clustered ols)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_boost'].transform('sum'),
        label='Predicted (boosting)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_lstm'].transform('sum'),
        label='Predicted (lstm)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_rf_et'].transform('sum'),
        label='Predicted (ensemble)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_avg'].transform('sum'),
        label='Predicted (avg)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_bates_granger'].transform('sum'),
        label='Predicted (bates granger)')
ax.plot(dfData['date'],
        dfData.groupby('date')['predicted_mse'].transform('sum'),
        label='Predicted (mse)')
ax.set_xlabel('Date')
ax.set_ylabel('Total Contribution')
ax.set_title('Full Sample')
ax.set_aspect('auto')
ax.set_ylim([-20, 100.00])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/FullSample/5_2_avg_fs.png")
plt.savefig("./Results/Presentation/FullSample/5_2_avg_fs.svg")
upload(plt, 'Project-based Internship', 'figures/5_2_avg_fs.png')

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

upload(dfRMSE_latex.to_latex(), 'Project-based Internship', 'tables/5_1_rmse.tex')

plt.close('all')

dfRMSE.to_csv("./Results/Tables/5_1_rmse.csv")

# Save to .parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")
dfDataWIP.to_parquet("./dfDataWIP_pred.parquet")

########################################################################################################################

dfDesc = pd.read_parquet('./.AUX/dfDesc.parquet')
dfData_org = pd.read_parquet('./dfData_org.parquet')

lJob = ['S218705', 'S100762', 'S289834', 'S102941']

plt.close('all')

# Create a subplot for each job_no in lJob
fig, ax = plt.subplots(len(lJob), 1, figsize=(20, 10 * len(lJob)))
# Loop through each job_no in lJob
for i, sJobNo in enumerate(lJob):
    # Plot total contribution, contribution, revenue and cumulative contribution
    ax[i].plot(dfData[dfData['job_no'] == sJobNo]['date'],
               dfData_org[dfData_org['job_no'] == sJobNo]['contribution_cumsum'],
               label='cumulative contribution')
    ax[i].plot(dfData[dfData['job_no'] == sJobNo]['date'],
               dfData_org[dfData_org['job_no'] == sJobNo]['final_estimate_contribution'],
               label='slutvurdering')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_lstm'],
               label='predicted (lstm)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_rf_et'],
               label='predicted (ensemble)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_mse'],
               label='predicted (mse)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_boost'],
               label='predicted (boosting)', linestyle='dashed')
    ax[i].axhline(y=0, color='black', linestyle='-')
    ax[i].set_xlabel('Date')
    ax[i].set_ylabel('Contribution (mDKK)')
    ax[i].set_title(f'Contribution of {sJobNo} - {dfDesc[dfDesc["job_no"] == sJobNo]["description"].values[0]}')
    ax[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.35)
    plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/Jobs/dl.png")


########################################################################################################################
# if ./Results/Figures/Jobs does not exist, create it

if not os.path.exists('./Results/Figures/Jobs'):
    os.makedirs('./Results/Figures/Jobs')

# For each job_no plot the actual and predicted sDepVar
for job_no in dfDataPred['job_no'].unique():
    # Get the data of job_no
    dfDataJob = dfDataPred[dfDataPred['job_no'] == job_no]
    # Plot the cumsum of actual and predicted contribution of sJobNo
    fig, ax = plt.subplots(figsize=(20, 10))
    for col in ['contribution_cumsum',
                'production_estimate_contribution',
                'predicted_boost',
                'final_estimate_contribution']:
        if col == 'production_estimate_contribution':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dotted')
        elif col == 'final_estimate_contribution':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dotted')
        elif col == 'predicted_lstm' or col == 'predicted_boost':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashdot')
        elif col == 'contribution_cumsum':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashed')
        else:
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution')
    ax.set_title(f'Actual vs. Predicted Total Contribution of {job_no}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.35)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/Jobs/{job_no}.png")
    plt.close('all')

########################################################################################################################


########################################################################################################################
# if ./Results/Figures/Jobs does not exist, create it
if not os.path.exists('./Results/Figures/WIP'):
    os.makedirs('./Results/Figures/WIP')
## For each job_no plot the actual and predicted sDepVar
for job_no in dfDataWIP['job_no'].unique():
    # Get the data of job_no
    dfDataJob = dfDataWIP[dfDataWIP['job_no'] == job_no]
    dfDataJob['LSTM'] = pd.DataFrame(
        model_fit.predict(
            dfDataJob[lNumericCols][dfDataJob[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
            batch_size=best_batch_size,
            use_multiprocessing=True, workers=multiprocessing.cpu_count()
        )
    )
    # Rescale
    dfDataJob["LSTM"] = y_scaler.inverse_transform(dfDataJob["LSTM"].values.reshape(-1, 1))
    # Plot the cumsum of actual and predicted contribution of sJobNo
    fig, ax = plt.subplots(figsize=(20, 10))
    for col in ['production_estimate_contribution',
                'predicted_boost',
                'predicted_lstm',
                'final_estimate_contribution',
                'contribution_cumsum']:
        if col == 'production_estimate_contribution':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dotted')
        elif col == 'final_estimate_contribution':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dotted')
        elif col == 'predicted_lstm':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashdot')
        elif col == 'contribution_cumsum':
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col, linestyle='dashed')
        else:
            ax.plot(dfDataJob['date'], dfDataJob[col], label=col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Contribution (mDKK)')
    ax.set_title(f'Actual vs. Predicted Total Contribution of {job_no}')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5).get_frame().set_linewidth(0.0)
    plt.grid(alpha=0.35)
    plt.rcParams['axes.axisbelow'] = True
    plt.savefig(f"./Results/Figures/WIP/{job_no}.png")
    plt.close('all')

########################################################################################################################


notify('The script has finished running.')