# Import required libraries
import runpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from pandas import DataFrame
from scipy.spatial import distance
import os
from matplotlib import rc
from plot_config import *

# Read data
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
# sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
os.chdir(sDir)

# Read dfData parquet file
dfData = pd.read_parquet("dfData.parquet")

dfData['date'] = pd.to_datetime(dfData['date'], format='%d-%m-%Y')
dfData['end_date'] = pd.to_datetime(dfData['end_date'], format='%d-%m-%Y')

# Plot distribution of budget_costs, sales_estimate_costs, production_estimate_costs and final_estimate_costs
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dfData, x='budget_costs', label='budget costs')
sns.kdeplot(data=dfData, x='sales_estimate_costs', label='sales estimate costs')
sns.kdeplot(data=dfData, x='production_estimate_costs', label='production estimate costs')
sns.kdeplot(data=dfData, x='final_estimate_costs', label='final estimate costs')
plt.xlabel("Costs (mDKK)")
plt.ylabel("Density")
plt.xlim(dfData['budget_costs'].quantile(0.00000000001), 10)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.25),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/1_0_costs.png")
plt.savefig("./Results/Presentation/1_0_costs.svg")
plt.show()

# Plot distribution of budget_revenue, sales_estimate_revenue, production_estimate_revenue and final_estimate_revenue
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dfData, x='budget_revenue', label='budget revenue')
sns.kdeplot(data=dfData, x='sales_estimate_revenue', label='sales estimate revenue')
sns.kdeplot(data=dfData, x='production_estimate_revenue', label='production estimate revenue')
sns.kdeplot(data=dfData, x='final_estimate_revenue', label='final estimate revenue')
# limit x-axis to cover 99.99% of the data
plt.xlim(-10, dfData['budget_revenue'].quantile(0.9999999999))
plt.xlabel("Revenue (mDKK)")
plt.ylabel("Density")
# legend below x-axis label
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.25),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/1_1_revenue.png")
plt.savefig("./Results/Presentation/1_1_revenue.svg")
plt.show()

# Plot sum of risk by date for each department
plt.figure(figsize=(10, 5))
sns.lineplot(x='date', y='risk', hue='department', data=dfData)
plt.xlabel("Date")
plt.ylabel("Risk")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.15),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/1_2_risk.png")
plt.savefig("./Results/Presentation/1_2_risk.svg")
plt.show()

# Select random job and plot risk
job_no = 'S161210'
dfData[dfData['job_no'] == job_no].plot(x='date', y='risk', figsize=(10, 5))
plt.xlabel("Date")
plt.ylabel("Risk")
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.15),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.show()

# Plot kde of risk
# Plot distribution of budget_costs, sales_estimate_costs, production_estimate_costs and final_estimate_costs
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dfData, x='risk', label='risk')
plt.xlabel("Risk (mDKK)")
plt.ylabel("Density")
plt.xlim(dfData['risk'].quantile(0.01), dfData['risk'].quantile(0.99))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.25),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/1_3_risk.png")
plt.savefig("./Results/Presentation/1_3_risk.svg")
plt.show()

### Missing Data Analysis ###
# Calculate missing values
dfMissing = dfData.isna().sum().reset_index()
dfMissing.columns = ['column', 'missing']

# Calculate missing percentage
dfMissing['missing_pct'] = dfMissing['missing'] / dfData.shape[0]

# Sort by missing percentage
dfMissing.sort_values('missing_pct', ascending=False, inplace=True)

# Plot missing percentage above 0%
plt.figure(figsize=(10, 5))
sns.barplot(x=dfMissing[dfMissing['missing_pct'] > 0]['column'],
            y=dfMissing[dfMissing['missing_pct'] > 0]['missing_pct'])
plt.xticks(rotation=90)
plt.xlabel("Columns")
plt.ylabel("Missing Percentage")
plt.tight_layout()
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.9),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/1_4_missing.png")
plt.savefig("./Results/Presentation/1_4_missing.svg")
plt.show()

# Plot kde of labor_cost_share, material_cost_share and other_cost_share
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dfData, x='labor_cost_share', label='labor cost share')
sns.kdeplot(data=dfData, x='material_cost_share', label='material cost share')
plt.xlabel("Share")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3).get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
                xy=(1.0, -0.25),
                color='grey',
                xycoords='axes fraction',
                ha='right',
                va="center",
                fontsize=10)
plt.savefig("./Results/Figures/1_5_cost_share.png")
plt.savefig("./Results/Presentation/1_5_cost_share.svg")
plt.show()

# Summary of Variables (mean, std, min, max, missing, % missing)
summary_data = dfData.describe().transpose()
# Format all numerical values in DataFrame with thousands separator.
# Keep index, min, max, mean, std.
formatted_df_eda_1 = summary_data[['mean', 'std', 'min', 'max']]
# Count number of missing values for each variable
missing_values = dfData.isnull().sum().to_frame()
# Rename column
missing_values = missing_values.rename(columns={0: 'missing'})
# Percentage of missing values
missing_values['% missing'] = missing_values['missing'] / len(dfData) * 100
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

eda_1 = eda_1.replace('\\begin{longtable}', '\\begin{landscape}\\begin{longtable}')
eda_1 = eda_1.replace('\\end{longtable}', '\\end{longtable}\\end{landscape}')

# Output to LaTeX with encoding
with open('Results/Tables/2_eda_1.tex', 'w', encoding='utf-8') as f:
    f.write(eda_1)
