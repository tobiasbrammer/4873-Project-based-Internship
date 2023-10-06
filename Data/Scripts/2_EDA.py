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
dfData['end_date'] = pd.to_datetime(dfData['end_date'], format='%Y-%m-%d')

# Summary of Data
# Format date and end_date as dd-mm-yyyy
dfData['date'] = dfData['date'].dt.strftime('%d-%m-%Y')
dfData['end_date'] = dfData['end_date'].dt.strftime('%d-%m-%Y')
summary_data = dfData.describe().transpose()
# Format all numerical values in DataFrame with thousands separator.
formatted_df_eda_1 = summary_data.map(lambda x: '{:,.2f}'.format(x) if isinstance(x, (int, float)) else x)
# Keep index, min, max, mean, std.
formatted_df_eda_1 = formatted_df_eda_1[['mean', 'std', 'min', 'max']]
# Count number of missing values for each variable
missing_values = dfData.isnull().sum().to_frame()
# Rename column
missing_values = missing_values.rename(columns={0: 'Missing'})
# Percentage of missing values
missing_values['% Missing'] = missing_values['Missing'] / len(dfData) * 100
# Add missing values to formatted_df_eda_1
formatted_df_eda_1 = formatted_df_eda_1.join(missing_values)

# Output to LaTeX with landscape orientation

eda_1 = formatted_df_eda_1.to_latex(index=True,
                                    caption='Variables',
                                    longtable=True,
                                    bold_rows=True,
                                    escape=True,
                                    label='eda_1').replace('%', '\\%')
# Make table landscape orientation (replace \begin{longtable} with \begin{landscape}\begin{longtable})
eda_1 = eda_1.replace('\\begin{longtable}', '\\begin{landscape}\\begin{longtable}')
eda_1 = eda_1.replace('\\end{longtable}', '\\end{longtable}\\end{landscape}')

# Output to LaTeX with landscape orientation
with open(f"{sDir}/eda_1.tex", "w") as f:
    f.write(eda_1)
