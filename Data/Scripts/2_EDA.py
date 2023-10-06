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
# sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship /Data"

os.chdir(sDir)
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")

# Convert the 'date' column to date format (assuming it's in 'YYYY-MM-DD' format)
dfData['date'] = pd.to_datetime(dfData['date'], format='%Y-%m-%d')

# Summary of Data
summary_data = dfData.describe().transpose()
# Format all numerical values in DataFrame with thousands separator
formatted_df = summary_data.applymap(lambda x: '{:,.2f}'.format(x) if isinstance(x, (int, float)) else x)
# Replace all percent signs % with \%
latex_str = formatted_df.style.to_latex().replace('%', '\\%')

# Save as LaTeX using Styler
with open('./Results/Tables/2_eda_1.tex', 'w', encoding='utf-8') as f:
    f.write(latex_str)
