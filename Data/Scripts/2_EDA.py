# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame
from scipy.spatial import distance
import os
from matplotlib import rc

# Read data
sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
# sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")

# Convert the 'date' column to date format (assuming it's in 'YYYY-MM-DD' format)
dfData['date'] = pd.to_datetime(dfData['date'], format='%Y-%m-%d')

# Summary of Data
summary_data = dfData.describe().transpose()
# Format all numerical values in DataFrame with thousands separator.
formatted_df_eda_1 = summary_data.map(lambda x: '{:,.2f}'.format(x) if isinstance(x, (int, float)) else x)
# Replace all percent signs % with \%
eda_1 = formatted_df_eda_1.to_latex(index=False, caption='Variables',longtable=True,label='eda_1').replace('%', '\\%')

eda_1 = eda_1.replace('NaN', 'Missing')

# Save as LaTeX using Styler
with open('./Results/Tables/2_eda_1.tex', 'w', encoding='utf-8') as f:
    f.write(eda_1)
