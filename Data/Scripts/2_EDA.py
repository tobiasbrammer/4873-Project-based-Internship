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
# sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")

dfData['date'] = pd.to_datetime(dfData['date'], format='%d-%m-%Y')
dfData['end_date'] = pd.to_datetime(dfData['end_date'], format='%d-%m-%Y')

# Summary of Variables (mean, std, min, max, missing, % missing)
summary_data = dfData.describe().transpose()
# Format all numerical values in DataFrame with thousands separator.
# Keep index, min, max, mean, std.
formatted_df_eda_1 = summary_data[['mean', 'std', 'min', 'max']]
# Upper case column names
formatted_df_eda_1.columns = formatted_df_eda_1.columns.str.upper()
# Count number of missing values for each variable
missing_values = dfData.isnull().sum().to_frame()
# Rename column
missing_values = missing_values.rename(columns={0: 'Missing'})
# Percentage of missing values
missing_values['% missing'] = missing_values['Missing'] / len(dfData) * 100
# Add missing values to formatted_df_eda_1
formatted_df_eda_1 = formatted_df_eda_1.join(missing_values)
# Format to show two decimals and use thousands separator
formatted_df_eda_1 = formatted_df_eda_1.round(2).applymap('{:,.2f}'.format)

# Output to LaTeX with landscape orientation
eda_1 = formatted_df_eda_1.style.to_latex(
    caption='List of Variables',
    position='h!',
    hrules=True,
    environment='longtable',
    label='eda_1').replace('%', '\\%')


eda_1 = eda_1.replace('\begin{longtable}', '\\begin{landscape}\\begin{longtable}')
eda_1 = eda_1.replace('\\end{longtable}', '\\end{longtable}\\end{landscape}')

# Output to LaTeX with encoding to show æ,ø,å
with open('Results/Tables/2_eda_1.tex', 'w', encoding='utf-8') as f:
    f.write(eda_1)


# Perform Shapiro-Wilk test for normality
# Create empty DataFrame
shapiro_test = pd.DataFrame()
# Loop through all numerical variables in dfData
for i in dfData.select_dtypes(include=np.number).columns:
    # Perform Shapiro-Wilk test
    stat, p = stats.shapiro(dfData[i])
    # Append results to DataFrame Use pandas.concat instead.
    shapiro_test = shapiro_test.append(pd.DataFrame({'Variable': [i],
                                                     'W': [stat],
                                                     'p-value': [p]}),
                                       ignore_index=True)

