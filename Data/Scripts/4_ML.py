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

# Load ./dfDataTest.parquet and ./dfDataTrain.parquet
dfDataScaled = pd.read_parquet("./dfDataScaled.parquet")
dfDataTest = pd.read_parquet("./dfDataTest.parquet")
dfDataTrain = pd.read_parquet("./dfDataTrain.parquet")

# Import scales
x_scaler = joblib.load("./.AUX/x_scaler.save")
y_scaler = joblib.load("./.AUX/y_scaler.save")

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
parametersGrid = {"max_iter": [1, 5, 10, 20, 50, 100, 200, 500, 1000],
                  "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  "l1_ratio": np.arange(0.0, 1.0, 0.1)}
elastic_net_cv = GridSearchCV(elastic_net, parametersGrid, cv=5, scoring='neg_mean_squared_error', verbose=1)

# Fit to the training data
elastic_net_cv.fit(dfDataTrain[lIndepVar_lag_budget], dfDataTrain[sDepVar])

# Predict on the test data
dfDataTrain['predicted_en'] = elastic_net_cv.predict(dfDataTrain[lIndepVar_lag_budget])
dfDataTest['predicted_en'] = elastic_net_cv.predict(dfDataTest[lIndepVar_lag_budget])

# Inverse transform
dfDataTrain['predicted_en'] = y_scaler.inverse_transform(dfDataTrain['predicted_en'].values.reshape(-1, 1))
dfDataTest['predicted_en'] = y_scaler.inverse_transform(dfDataTest['predicted_en'].values.reshape(-1, 1))


dfDataTrain['sum_predicted_es'] = dfDataTrain.groupby('date')['predicted_es'].transform('sum')
dfDataTest['sum_predicted_es'] = dfDataTest.groupby('date')['predicted_es'].transform('sum')

# Calculate RMSE
rmse_en = np.sqrt(mean_squared_error(dfDataTest[sDepVar], dfDataTest['predicted_en']))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dfDataTest['date'], dfDataTest['sum'], label='Actual')
ax.plot(dfDataTest['date'], dfDataTest['sum_predicted_es'], label='Predicted')
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
plt.savefig("./Results/Figures/3_5_es.png")
plt.savefig("./Results/Presentation/3_5_es.svg")

