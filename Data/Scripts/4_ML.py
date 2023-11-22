# Import required libraries
import warnings
import os
import numpy as np
import pandas as pd
import datetime
import joblib
from plot_config import *
from plot_predicted import *
from smape import *
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

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

# Load dfDataPred from ./dfDataPred.parquet
dfDataPred = pd.read_parquet("./dfDataPred.parquet")

# Load dfDataWIP from ./dfDataWIP_pred.parquet
dfDataWIP = pd.read_parquet("./dfDataWIP_pred.parquet")


# Load trainMethod from ./.AUX/trainMethod.txt
with open('./.AUX/trainMethod.txt', 'r') as f:
    trainMethod = f.read()

# Import lNumericCols from ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'r') as f:
    lNumericCols = f.read()
lNumericCols = lNumericCols.split('\n')

# Omit columns that contain 'cluster_'
lNumericCols = [col for col in lNumericCols if not col.startswith('cluster_')]

# Replace infinite values with NaN
dfDataScaled.replace([np.inf, -np.inf], np.nan, inplace=True)
dfData.replace([np.inf, -np.inf], np.nan, inplace=True)

# If trainMethod == 'train', then oTrain = 'train_TS' else oTrain = 'train'
if trainMethod == 'train':
    oTrain = 'train_TS'
else:
    oTrain = 'train'


# Replace NaN with 0
dfDataScaled.select_dtypes(include=['float64', 'int64']).fillna(0, inplace=True)
dfData.select_dtypes(include=['float64', 'int64']).fillna(0, inplace=True)

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

# Import lDST from ./.AUX/lDST.txt
with open('./.AUX/lDST.txt', 'r') as f:
    lDST = f.read()
lDST = lDST.split('\n')

# Rescale dfDataScaled to dfData
dfDataRescaled = dfDataScaled.copy()
dfDataRescaled[colIndepVarNum] = x_scaler.inverse_transform(dfDataScaled[colIndepVarNum].values)
dfDataRescaled[sDepVar] = y_scaler.inverse_transform(dfDataScaled[sDepVar].values.reshape(-1, 1))

train_index = dfData[dfData[trainMethod] == 1].index
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
# Define hyperparameter grid
param_grid_en = {
    'alpha': np.arange(0.1, 10, 0.01),
    'l1_ratio': np.arange(0.0000001, 1.00, 0.0001),
    'tol': np.arange(0.00001, 0.50, 0.01)
}
# Define randomized search
elastic_net = ElasticNet(tol=1e-4, random_state=0)
elastic_net_cv = RandomizedSearchCV(elastic_net, param_grid_en, n_iter=1000, scoring=None, cv=3, verbose=0,
                                           refit=True, n_jobs=-1, random_state=0,
                                           return_train_score=True)

# get unique entries in lIndepVar_lag_budget + lDST
lIndepVar = list(set(lIndepVar_lag_budget)) + list(set(lDST))
# Drop duplicates from lIndepVar
lIndepVar = list(dict.fromkeys(lIndepVar))
# If lIndepVar contains like 'cluster_' then remove it
lIndepVar = [col for col in lIndepVar if not col.startswith('cluster_')]

# Save lIndepVar to ./.AUX/lIndepVar.txt
with open('./.AUX/lIndepVar.txt', 'w') as f:
    f.write('\n'.join(lIndepVar))

lIndepVar = lIndepVar + ['intercept']
# Sparse model with OLS variables
start_time_en_sparse = datetime.datetime.now()
elastic_net_cv.fit(dfDataScaledTrain[lIndepVar].replace(np.nan, 0), dfDataScaledTrain[sDepVar])

# Get more precise hyperparamters list((np.arange(0.8, 1.2, 0.05)*gb_cv.best_params_.get('learning_rate')).round(2)),
param_grid_en_detail = {
    'alpha': list((np.arange(0.8, 1.2, 0.01)*elastic_net_cv.best_params_.get('alpha')).round(4)),
    'l1_ratio': list((np.arange(0.8, 1.2, 0.01)*elastic_net_cv.best_params_.get('l1_ratio')).round(4)),
    'tol': list((np.arange(0.8, 1.2, 0.01)*elastic_net_cv.best_params_.get('tol')).round(4))
}
elastic_net_cv = RandomizedSearchCV(elastic_net, param_grid_en_detail, n_iter=1000, scoring=None, cv=3, verbose=0,
                                           refit=True, n_jobs=-1)
elastic_net_cv.fit(dfDataScaledTrain[lIndepVar].replace(np.nan, 0), dfDataScaledTrain[sDepVar])

# Get best hyperparameters
print(f'The optimal EN alpha is {elastic_net_cv.best_params_.get("alpha")}.')
print(f'The optimal EN l1_ratio is {elastic_net_cv.best_params_.get("l1_ratio")}.')
print(f'The optimal EN tol is {elastic_net_cv.best_params_.get("tol")}.')

# Save model to .MODS/ as pickle
joblib.dump(elastic_net_cv, './.MODS/elastic_net_cv.pickle')

dfData['predicted_en'] = elastic_net_cv.predict(dfDataScaled[lIndepVar].replace(np.nan, 0))
dfData['predicted_en'] = y_scaler.inverse_transform(dfData['predicted_en'].values.reshape(-1, 1))
end_time_en_sparse = datetime.datetime.now()
print(f'ElasticNet finished in {end_time_en_sparse - start_time_en_sparse}.')

plot_predicted(dfData, 'predicted_en', 'Elastic Net', '4_0_en', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of EN
rmse_en_sparse = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_en'].replace(np.nan, 0)))
# Calculate sMAPE
smape_en_sparse = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                        dfData[dfData[trainMethod] == 0]['predicted_en'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Elastic Net', 'RMSE'] = rmse_en_sparse
dfRMSE.loc['Elastic Net', 'sMAPE'] = smape_en_sparse

# Add to dfDataPred
dfDataPred['predicted_en'] = dfData['predicted_en']

# Predict WIP
dfDataWIP['predicted_en'] = elastic_net_cv.predict(dfDataWIP[lIndepVar].replace(np.nan, 0))
dfDataWIP['predicted_en'] = y_scaler.inverse_transform(dfDataWIP['predicted_en'].values.reshape(-1, 1))

### Random Forest Regression ###
# Define Random Forest model
rf = RandomForestRegressor(n_jobs=-1, random_state=0, verbose=False)

## Define hyperparameter grid ##
rf_grid = {
    "n_estimators": np.arange(10, 300, 5),  # Number of trees
    "max_depth": [1, 3, 5, 10, 15, 20, 25, 50, 75, 100],  # Depth of each tree
    "min_samples_split": np.arange(1, 60, 1),  # Minimum samples required to split an internal node
    "min_samples_leaf": np.arange(1, 60, 1),  # Minimum samples required to be at a leaf node
    "max_features": [1 / 3, 0.5, 1, "sqrt", "log2"],  # Number of features to consider for split
    "max_samples": [50, 100, 150, 250, 500, 750, 1000, 1500]  # Number of samples to train each tree
}

# Define randomized search
rf_cv = RandomizedSearchCV(rf, rf_grid, n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error", cv=3, verbose=False,
                           refit=True)
# Fit to the training data
start_time_rf = datetime.datetime.now()
rf_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
          dfDataScaledTrain[sDepVar])

rf_grid_detail = {
    "n_estimators": list((np.arange(0.8, 1.2, 0.05)*rf_cv.best_params_.get('n_estimators')).astype('int')),
    "max_depth": list((np.arange(0.8, 1.2, 0.05)*rf_cv.best_params_.get('max_depth')).astype('int')),
    "min_samples_split": list((np.arange(0.8, 1.2, 0.05)*rf_cv.best_params_.get('min_samples_split')).astype('int')),
    "min_samples_leaf": list((np.arange(0.8, 1.2, 0.05)*rf_cv.best_params_.get('min_samples_leaf')).astype('int')),
    "max_features": rf_grid.get('max_features'),
    "max_samples": list((np.arange(0.8, 1.2, 0.05)*rf_cv.best_params_.get('max_samples')).astype('int'))
}

rf_cv = RandomizedSearchCV(rf, rf_grid_detail, n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error", cv=3, verbose=False,
                           refit=True)
rf_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
          dfDataScaledTrain[sDepVar])

# Save model to .MODS/ as pickle
joblib.dump(rf_cv, './.MODS/rf_cv.pickle')

# Predict and rescale using RF
dfData['predicted_rf_full'] = rf_cv.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_rf_full'] = y_scaler.inverse_transform(dfData['predicted_rf_full'].values.reshape(-1, 1))

end_time_rf = datetime.datetime.now()

print(f'     ')
print(f'RF Full fit finished in {end_time_rf - start_time_rf}.')
print(f'     ')
# Print hyperparameters
print(f'The optimal RF number of estimators is {rf_cv.best_params_.get("n_estimators")}.')
print(f'The optimal RF maximum depth is {rf_cv.best_params_.get("max_depth").astype("int")}.')
print(f'The optimal RF minimum sample split is {rf_cv.best_params_.get("min_samples_split").astype("int")}.')
print(f'The optimal RF minimum sample leaf is {rf_cv.best_params_.get("min_samples_leaf").astype("int")}.')
print(f'The optimal RF maximum features is {rf_cv.best_params_.get("max_features")}.')
print(f'The optimal RF maximum samples is {rf_cv.best_params_.get("max_samples").astype("int")}.')
print(f'The optimal RF RMSE is {np.sqrt(-rf_cv.best_score_).round(4)}.')

plot_predicted(dfData, 'predicted_rf_full', 'Random Forest', '4_1_rf_full', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of RF
rmse_rf_full = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_rf_full'].replace(np.nan, 0)))
# Calculate sMAPE
smape_rf_full = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfData[dfData[trainMethod] == 0]['predicted_rf_full'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Random Forest (Full)', 'RMSE'] = rmse_rf_full
dfRMSE.loc['Random Forest (Full)', 'sMAPE'] = smape_rf_full

# Add to dfDataPred
dfDataPred['predicted_rf_full'] = dfData['predicted_rf_full']

# Predict WIP
dfDataWIP['predicted_rf_full'] = rf_cv.predict(dfDataWIP[lNumericCols][dfDataWIP[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfDataWIP['predicted_rf_full'] = y_scaler.inverse_transform(dfDataWIP['predicted_rf_full'].values.reshape(-1, 1))

# Variable importance
dfVarImp = pd.DataFrame(rf_cv.best_estimator_.feature_importances_, index=[dfDataWIP[lNumericCols].columns.difference([sDepVar])], columns=['importance'])
dfVarImp.drop('intercept', inplace=True)
dfVarImp = dfVarImp[dfVarImp['importance'] > 0.01]
dfVarImp.sort_values(by='importance', ascending=False, inplace=True)
dfVarImp['cumsum'] = dfVarImp['importance'].cumsum()
dfVarImp['cumsum'] = dfVarImp['cumsum'] / dfVarImp['cumsum'].max()

print(f'     ')
print(f'Variable importance of Random Forest:')
print(dfVarImp)


# Random Forest using only lIndepVar
# Define randomized search
rf_cv = RandomizedSearchCV(rf, rf_grid_detail, n_iter=100, n_jobs=-1, scoring="neg_mean_squared_error", cv=3, verbose=False,
                           refit=True)
# Fit to the training data
start_time_rf = datetime.datetime.now()
rf_cv.fit(dfDataScaledTrain[lIndepVar][dfDataScaledTrain[lIndepVar].columns.difference([sDepVar])],
          dfDataScaledTrain[sDepVar])
# Save model to .MODS/ as pickle
joblib.dump(rf_cv, './.MODS/rf_cv_sparse.pickle')
# Predict and rescale using RF
dfData['predicted_rf_sparse'] = rf_cv.predict(
    dfDataScaled[lIndepVar][dfDataScaled[lIndepVar].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_rf_sparse'] = y_scaler.inverse_transform(dfData['predicted_rf_sparse'].values.reshape(-1, 1))
end_time_rf = datetime.datetime.now()
print(f'     ')
print(f'RF Sparse fit finished in {end_time_rf - start_time_rf}.')

plot_predicted(dfData, 'predicted_rf_sparse', 'Random Forest', '4_2_rf_sparse', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of RF
rmse_rf_sparse = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfData[dfData[trainMethod] == 0]['predicted_rf_sparse'].replace(np.nan, 0)))
# Calculate sMAPE
smape_rf_sparse = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                        dfData[dfData[trainMethod] == 0]['predicted_rf_sparse'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Random Forest (Sparse)', 'RMSE'] = rmse_rf_sparse
dfRMSE.loc['Random Forest (Sparse)', 'sMAPE'] = smape_rf_sparse

# Add to dfDataPred
dfDataPred['predicted_rf_sparse'] = dfData['predicted_rf_sparse']

# Predict WIP
dfDataWIP['predicted_rf_sparse'] = rf_cv.predict(
    dfDataWIP[lIndepVar][dfDataWIP[lIndepVar].columns.difference([sDepVar])].replace(np.nan, 0))
dfDataWIP['predicted_rf_sparse'] = y_scaler.inverse_transform(dfDataWIP['predicted_rf_sparse'].values.reshape(-1, 1))

## Extremely Randomized Trees ##
# Define Extremely Randomized Trees model
from sklearn.ensemble import ExtraTreesRegressor

# Define grid for et
et_grid = {
    "n_estimators": np.arange(10, 300, 5),  # Number of trees
    "max_depth": [1, 3, 5, 10, 15, 20, 25, 50, 75, 100],  # Depth of each tree
    "min_samples_split": np.arange(1, 60, 1),  # Minimum samples required to split an internal node
    "min_samples_leaf": np.arange(1, 60, 1),  # Minimum samples required to be at a leaf node
    "max_features": [1 / 3, 0.5, 1, "sqrt", "log2"],  # Number of features to consider for split
    "bootstrap": [True, True, True, True, True], # Whether bootstrap samples are used when building trees.
    "max_samples": [50, 100, 150, 250, 500, 750, 1000, 1500]  # Number of samples to train each tree
}

# Define randomized search
et_cv = RandomizedSearchCV(ExtraTreesRegressor(random_state=0), et_grid, n_iter=100, n_jobs=-1,
                            scoring="neg_mean_squared_error", cv=3, verbose=False, refit=True,
                           )
# Fit to the training data
start_time_et = datetime.datetime.now()


# Detailed grid
et_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
            dfDataScaledTrain[sDepVar])

et_grid_detail = {
    "n_estimators": list((np.arange(0.8, 1.2, 0.05)*et_cv.best_params_.get('n_estimators')).astype('int')),
    "max_depth": list((np.arange(0.8, 1.2, 0.05)*et_cv.best_params_.get('max_depth')).astype('int')),
    "min_samples_split": list((np.arange(0.8, 1.2, 0.05)*et_cv.best_params_.get('min_samples_split')).astype('int')),
    "min_samples_leaf": list((np.arange(0.8, 1.2, 0.05)*et_cv.best_params_.get('min_samples_leaf')).astype('int')),
    "max_features": et_grid.get('max_features'),
    "bootstrap": et_grid.get('bootstrap'),
    "max_samples": list((np.arange(0.8, 1.2, 0.05)*et_cv.best_params_.get('max_samples')).astype('int'))
}

et_cv = RandomizedSearchCV(ExtraTreesRegressor(random_state=0), et_grid_detail, n_iter=100, n_jobs=-1,
                            scoring="neg_mean_squared_error", cv=3, verbose=False, refit=True,
                           )

et_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
            dfDataScaledTrain[sDepVar])

# Save model to .MODS/ as pickle
joblib.dump(et_cv, './.MODS/et_cv.pickle')
# Predict and rescale using ET
dfData['predicted_et'] = et_cv.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_et'] = y_scaler.inverse_transform(dfData['predicted_et'].values.reshape(-1, 1))

print(f'     ')
print(f'ET fit finished in {datetime.datetime.now() - start_time_et}.')
print(f'     ')
# Print hyperparameters
print(f'The optimal ET number of estimators is {et_cv.best_params_.get("n_estimators")}.')
print(f'The optimal ET maximum depth is {et_cv.best_params_.get("max_depth")}.')
print(f'The optimal ET minimum sample split is {et_cv.best_params_.get("min_samples_split").astype("int")}.')
print(f'The optimal ET minimum sample leaf is {et_cv.best_params_.get("min_samples_leaf").astype("int")}.')
print(f'The optimal ET maximum features is {et_cv.best_params_.get("max_features")}.')
print(f'The optimal ET maximum samples is {et_cv.best_params_.get("max_samples")}.')
print(f'The optimal ET RMSE is {np.sqrt(-et_cv.best_score_).round(4)}.')

plot_predicted(dfData, 'predicted_et', 'Extra Trees', '4_3_et', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of ET
rmse_et = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                          dfData[dfData[trainMethod] == 0]['predicted_et'].replace(np.nan, 0)))
# Calculate sMAPE
smape_et = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                        dfData[dfData[trainMethod] == 0]['predicted_et'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Extra Trees', 'RMSE'] = rmse_et
dfRMSE.loc['Extra Trees', 'sMAPE'] = smape_et

# Add to dfDataPred
dfDataPred['predicted_et'] = dfData['predicted_et']

# Predict WIP
dfDataWIP['predicted_et'] = et_cv.predict(
    dfDataWIP[lNumericCols][dfDataWIP[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))


### Boosted Regression Trees ###
# Define Boosted Regression Trees model
from sklearn.ensemble import GradientBoostingRegressor

# Set random grid for Gradient Boosting
gb_grid = {
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    'max_depth': [None, 1, 3, 5, 10, 15, 20, 25],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 12, 14],
    'max_features': [1 / 3, 0.5, 1, "sqrt", "log2", None],
    'n_estimators': [100, 150, 250, 500, 1000, 1500, 2000]
}

# Define randomized search
gb_cv = RandomizedSearchCV(GradientBoostingRegressor(random_state=0), gb_grid, n_iter=100, scoring=None, cv=3,
                           verbose=0, refit=True, n_jobs=-1)

# Fit to the training data
start_time_gb = datetime.datetime.now()
gb_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
          dfDataScaledTrain[sDepVar])

# Generate of sequence of numbers based on gb_cv.best_params_ to get more appropriate parameters in the defined range.
gb_grid_detail = {
    'learning_rate': list((np.arange(0.8, 1.2, 0.05)*gb_cv.best_params_.get('learning_rate')).round(4)),
    'max_depth': list((np.arange(0.8, 1.2, 0.05)*gb_cv.best_params_.get('max_depth')).astype("int")),
    'min_samples_leaf': list((np.arange(0.8, 1.2, 0.05)*gb_cv.best_params_.get('min_samples_leaf')).astype("int")),
    'max_features': gb_grid.get('max_features'),
    'n_estimators': list((np.arange(0.8, 1.2, 0.05)*gb_cv.best_params_.get('n_estimators')).astype("int")),
}

# Define randomized search
gb_cv_det = RandomizedSearchCV(GradientBoostingRegressor(random_state=0), gb_grid_detail, n_iter=100, scoring=None, cv=3,
                           verbose=0, refit=True, n_jobs=-1)
# Fit to the training data
gb_cv_det.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
          dfDataScaledTrain[sDepVar])

# Save model to .MODS/ as pickle
joblib.dump(gb_cv_det, './.MODS/gb_cv.pickle')

# Predict and rescale using GB
dfData['predicted_gb'] = gb_cv_det.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_gb'] = y_scaler.inverse_transform(dfData['predicted_gb'].values.reshape(-1, 1))

end_time_gb = datetime.datetime.now()

print(f'     ')
print(f'GB fit finished in {end_time_gb - start_time_gb}.')
print(f'     ')
# Optimal hyperparameters
print(f'The optimal GB learning rate is {gb_cv_det.best_params_.get("learning_rate")}.')
print(f'The optimal GB maximum depth is {gb_cv_det.best_params_.get("max_depth")}.')
print(f'The optimal GB minimum sample leaf is {gb_cv_det.best_params_.get("min_samples_leaf")}.')
print(f'The optimal GB maximum features is {gb_cv_det.best_params_.get("max_features")}.')
print(f'The optimal GB number of estimators is {gb_cv_det.best_params_.get("n_estimators")}.')

plot_predicted(dfData, 'predicted_gb', 'Gradient Boosting', '4_4_gb', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of GB
rmse_gb = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_gb'].replace(np.nan, 0)))
# Calculate sMAPE
smape_gb = smape(dfData[dfData[trainMethod] == 0][sDepVar], dfData[dfData[trainMethod] == 0]['predicted_gb'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Gradient Boosting', 'RMSE'] = rmse_gb
dfRMSE.loc['Gradient Boosting', 'sMAPE'] = smape_gb

# Add to dfDataPred
dfDataPred['predicted_gb'] = dfData['predicted_gb']

# Predict WIP
dfDataWIP['predicted_gb'] = gb_cv_det.predict(
    dfDataWIP[lNumericCols][dfDataWIP[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfDataWIP['predicted_gb'] = y_scaler.inverse_transform(dfDataWIP['predicted_gb'].values.reshape(-1, 1))


### XGBoost Regression ###
# Define XGBoost model
from xgboost import XGBRegressor

xgb_grid = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.001, 0.01, 0.1],
    'reg_lambda': [0, 0.001, 0.01, 0.1],
}

xgb_cv = RandomizedSearchCV(XGBRegressor(random_state=0), xgb_grid, n_iter=100, scoring=None, cv=3,
                            verbose=0, refit=True, n_jobs=-1)

# Fit to the training data
start_time_xgb = datetime.datetime.now()
xgb_cv.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
           dfDataScaledTrain[sDepVar])

# Detailed grid
xgb_grid_detail = {
    'learning_rate': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('learning_rate')).round(2)),
    'n_estimators': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('n_estimators')).astype("int")),
    'max_depth': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('max_depth')).astype("int")),
    'min_child_weight': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('min_child_weight')).round(2)),
    'gamma': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('gamma')).round(2)),
    'subsample': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('subsample')).round(2)),
    'colsample_bytree': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('colsample_bytree')).round(2)),
    'reg_alpha': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('reg_alpha')).round(4)),
    'reg_lambda': list((np.arange(0.8, 1.2, 0.05)*xgb_cv.best_params_.get('reg_lambda')).round(2))
}

xgb_cv_det = RandomizedSearchCV(XGBRegressor(random_state=0), xgb_grid_detail, n_iter=100, scoring=None, cv=3,
                            verbose=0, refit=True, n_jobs=-1)

xgb_cv_det.fit(dfDataScaledTrain[lNumericCols][dfDataScaledTrain[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0),
           dfDataScaledTrain[sDepVar])

# Save model to .MODS/ as pickle
joblib.dump(xgb_cv_det, './.MODS/xgb_cv.pickle')
# Predict and rescale using XGB
dfData['predicted_xgb'] = xgb_cv_det.predict(
    dfDataScaled[lNumericCols][dfDataScaled[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfData['predicted_xgb'] = y_scaler.inverse_transform(dfData['predicted_xgb'].values.reshape(-1, 1))
end_time_xgb = datetime.datetime.now()

print(f'     ')
print(f'XGB fit finished in {end_time_xgb - start_time_xgb}.')
print(f'     ')
# Best hyperparameters
print(f'Optimal learning rate: {xgb_cv_det.best_params_.get("learning_rate")}')
print(f'Optimal number of estimators: {xgb_cv_det.best_params_.get("n_estimators")}')
print(f'Optimal maximum depth: {xgb_cv_det.best_params_.get("max_depth")}')
print(f'Optimal minimum child weight: {xgb_cv_det.best_params_.get("min_child_weight")}')
print(f'Optimal gamma: {xgb_cv_det.best_params_.get("gamma")}')
print(f'Optimal subsample: {xgb_cv_det.best_params_.get("subsample")}')
print(f'Optimal colsample bytree: {xgb_cv_det.best_params_.get("colsample_bytree")}')
print(f'Optimal reg alpha: {xgb_cv_det.best_params_.get("reg_alpha")}')
print(f'Optimal reg lambda: {xgb_cv_det.best_params_.get("reg_lambda")}')

plot_predicted(dfData, 'predicted_xgb', 'XGBoost', '4_5_xgb', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)


# Calculate RMSE of XGB
rmse_xgb = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfData[dfData[trainMethod] == 0]['predicted_xgb'].replace(np.nan, 0)))
# Calculate sMAPE
smape_xgb = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfData[dfData[trainMethod] == 0]['predicted_xgb'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['XGBoost', 'RMSE'] = rmse_xgb
dfRMSE.loc['XGBoost', 'sMAPE'] = smape_xgb

# Add to dfDataPred
dfDataPred['predicted_xgb'] = dfData['predicted_xgb']

# Predict WIP
dfDataWIP['predicted_xgb'] = xgb_cv_det.predict(
    dfDataWIP[lNumericCols][dfDataWIP[lNumericCols].columns.difference([sDepVar])].replace(np.nan, 0))
dfDataWIP['predicted_xgb'] = y_scaler.inverse_transform(dfDataWIP['predicted_xgb'].values.reshape(-1, 1))


### Forecast Combination with Boosting
dfDataPred['predicted_boost'] = (dfDataPred['predicted_gb'] + dfDataPred['predicted_xgb']) / 2
dfData['predicted_boost'] = (dfData['predicted_gb'] + dfDataPred['predicted_xgb']) / 2
dfDataWIP['predicted_boost'] = (dfDataWIP['predicted_gb'] + dfDataWIP['predicted_xgb']) / 2

plot_predicted(dfData, 'predicted_boost', 'Combined Boosting', '4_6_boost', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)


# Calculate RMSE of GB_FC
rmse_gb_fc = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfDataPred[dfData[trainMethod] == 0]['predicted_boost'].replace(np.nan, 0)))
# Calculate sMAPE
smape_gb_fc = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfDataPred[dfData[trainMethod] == 0]['predicted_boost'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Combined Boosting', 'RMSE'] = rmse_gb_fc
dfRMSE.loc['Combined Boosting', 'sMAPE'] = smape_gb_fc

### Forecast Combination with RF and ET
dfDataPred['predicted_rf_et'] = (dfDataPred['predicted_rf_full'] + dfDataPred['predicted_et']) / 2
dfData['predicted_rf_et'] = (dfData['predicted_rf_full'] + dfDataPred['predicted_et']) / 2
dfDataWIP['predicted_rf_et'] = (dfDataWIP['predicted_rf_full'] + dfDataWIP['predicted_et']) / 2

plot_predicted(dfData, 'predicted_rf_et', 'Ensemble of Ensembles', '4_7_rf_et', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of GB_FC
rmse_rf_et = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                       dfDataPred[dfData[trainMethod] == 0]['predicted_rf_et'].replace(np.nan, 0)))
# Calculate sMAPE
smape_rf_et = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfDataPred[dfData[trainMethod] == 0]['predicted_rf_et'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Ensemble of Ensembles', 'RMSE'] = rmse_rf_et
dfRMSE.loc['Ensemble of Ensembles', 'sMAPE'] = smape_rf_et

## Average of rf, et, gb and xgb
dfDataPred['predicted_avg_ml'] = (dfDataPred['predicted_rf_sparse'] + dfDataPred['predicted_et'] + dfDataPred['predicted_gb'] + dfDataPred['predicted_xgb']) / 4
dfData['predicted_avg_ml'] = (dfData['predicted_rf_sparse'] + dfDataPred['predicted_et'] + dfDataPred['predicted_gb'] + dfDataPred['predicted_xgb']) / 4

plot_predicted(dfData, 'predicted_avg_ml', 'Average of Ensembles', '4_8_ml_avg', transformation='sum', trainMethod=trainMethod, sDepVar=sDepVar)

# Calculate RMSE of GB_FC
rmse_avg = np.sqrt(
    mean_squared_error(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0),
                          dfDataPred[dfData[trainMethod] == 0]['predicted_avg_ml'].replace(np.nan, 0)))
# Calculate sMAPE
smape_avg = smape(dfData[dfData[trainMethod] == 0][sDepVar].replace(np.nan, 0), dfDataPred[dfData[trainMethod] == 0]['predicted_avg_ml'].replace(np.nan, 0))

# Add to dfRMSE
dfRMSE.loc['Average of ML', 'RMSE'] = rmse_avg
dfRMSE.loc['Average of ML', 'sMAPE'] = smape_avg


# Save dfDataPred to ./dfDataPred.parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")
dfDataWIP.to_parquet("./dfDataWIP_pred.parquet")

# Round to 4 decimals
dfRMSE = dfRMSE.round(4)
print(dfRMSE)

dfRMSE.to_csv("./Results/Tables/3_4_rmse.csv")

plt.close('all')

########################################################################################################################

# Save to .parquet
dfDataPred.to_parquet("./dfDataPred.parquet")
dfData.to_parquet("./dfData_reg.parquet")

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
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_rf_et'],
               label='predicted (ensemble)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_avg_ml'],
               label='predicted (ml)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_en'],
               label='predicted (elastic net)', linestyle='dashed')
    ax[i].plot(dfDataPred[dfDataPred['job_no'] == sJobNo]['date'],
               dfDataPred[dfDataPred['job_no'] == sJobNo]['predicted_boost'],
               label='predicted (boost)', linestyle='dashed')
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
plt.savefig("./Results/Figures/Jobs/ml.png")
# Save figure
