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
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

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

# Import lNumericCols from ./.AUX/lNumericCols.txt
with open('./.AUX/lNumericCols.txt', 'r') as f:
    lNumericCols = f.read()
lNumericCols = lNumericCols.split('\n')

### Elastic Net Regression ###
# Define Elastic Net model
elastic_net = ElasticNet()

# Define hyperparameter grid
parametersGrid = {"max_iter": [10, 100, 200, 500, 1000, 2000, 5000, 10000],
                  "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  "l1_ratio": np.arange(0.01, 1.0, 0.01)}
elastic_net_cv = GridSearchCV(elastic_net, parametersGrid, cv=25, scoring='neg_mean_squared_error', verbose=0)

# Fit to the training data
elastic_net_cv.fit(dfDataScaledTrain[lIndepVar_lag_budget], dfDataScaledTrain[sDepVar])

# Predict and rescale using EN
dfData['predicted_en'] = elastic_net_cv.predict(dfDataScaled[lIndepVar_lag_budget])

dfData['predicted_en'] = y_scaler.inverse_transform(dfData['predicted_en'].values.reshape(-1, 1))

# Group by date and sum predicted_en
dfData['sum_predicted_en'] = dfData.groupby('date')['predicted_en'].transform('sum')

# Plot the sum of predicted and actual sDepVar by date
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfData['date'], dfData['sum'], label='Actual')
ax.plot(dfData['date'], dfData['sum_predicted_en'], label='Predicted (Elastic Net)')
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

# Print best hyperparameters
print(f"The best hyperparameters are: {elastic_net_cv.best_params_}")

# Print best score
print(f"The best score is: {elastic_net_cv.best_score_}")

