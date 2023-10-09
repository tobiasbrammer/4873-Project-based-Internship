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

# Load ./dfData.parquet
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(sDir)

# Predict dependent variable using PLS
# omit date and end_date
dfDataFinishedTrainIndep_date = dfDataFinishedTrainIndep['date']
# Only keep numeric columns
dfDataFinishedTrainIndep_2 = dfDataFinishedTrainIndep.select_dtypes(include=[np.number])

dfDataFinishedTrainIndep_2 = savgol_filter(dfDataFinishedTrainIndep_2, 12, 3)