# Import required libraries
import os
import pandas as pd
import numpy as np
from plot_config import *
from plot_predicted import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Turn off RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Read data
if os.name == 'posix':
    sDir = "/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship/Data"
# If operating system is Windows then
elif os.name == 'nt':
    sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

os.chdir(sDir)

import dropbox
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import re
import subprocess

# Read dfData parquet file
dfData = pd.read_parquet("dfData_org.parquet")

dfData['date'] = pd.to_datetime(dfData['date'], format='%d-%m-%Y')
dfData['end_date'] = pd.to_datetime(dfData['end_date'], format='%d-%m-%Y')

# Plot distribution of budget_costs, sales_estimate_costs, production_estimate_costs and final_estimate_costs
plt.figure(figsize=(20, 10))
sns.kdeplot(data=dfData, x='budget_costs', label='budget costs')
sns.kdeplot(data=dfData, x='sales_estimate_costs', label='sales estimate costs')
sns.kdeplot(data=dfData, x='production_estimate_costs', label='production estimate costs')
sns.kdeplot(data=dfData, x='final_estimate_costs', label='final estimate costs')
plt.rcParams.update({'font.size': 20})
# Set font size of x and y labels
plt.xlabel("Costs (mDKK)", fontsize=20)
plt.ylabel("Density", fontsize=20)
plt.xlim(dfData['budget_costs'].quantile(0.00000000001), 10)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/1_0_costs.png")
plt.savefig("./Results/Presentation/1_0_costs.svg")
upload(plt, 'Project-based Internship', 'figures/1_0_costs.png')

# Plot distribution of budget_revenue, sales_estimate_revenue, production_estimate_revenue and final_estimate_revenue
plt.figure(figsize=(20, 10))
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

plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/1_1_revenue.png")
plt.savefig("./Results/Presentation/1_1_revenue.svg")
upload(plt, 'Project-based Internship', 'figures/1_1_revenue.png')


# Select random job
job_no = 'S309436'
dfJob = dfData[dfData['job_no'] == job_no]

# Order by date
pd.options.mode.chained_assignment = None  # default='warn'
dfJob.sort_values('date', inplace=True)

fig, ax = plt.subplots(2, 1, figsize=(20, 10))
sns.lineplot(x='date', y='revenue_cumsum', data=dfJob, ax=ax[0], color=vColors[0], label='revenue')
sns.lineplot(x='date', y='revenue_scurve', data=dfJob, ax=ax[0], color=vColors[0], label='revenue_scurve', linestyle='--')
ax[0].set_xlabel("Date")
ax[0].set_ylabel("Revenue (mDKK)")
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
sns.lineplot(x='date', y='costs_cumsum', data=dfJob, ax=ax[1], color=vColors[1], label='costs')
sns.lineplot(x='date', y='costs_scurve', data=dfJob, ax=ax[1], color=vColors[1], label='costs_scurve', linestyle='--')
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Costs (mDKK)")
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.suptitle(f"Job {job_no}")
plt.savefig("./Results/Figures/1_2_scurve.png")
plt.savefig("./Results/Presentation/1_2_scurve.svg")
upload(plt, 'Project-based Internship', 'figures/1_2_scurve.png')

# Plot kde of risk and sum of risk by date for each department in a grid with 2 rows and 1 column
fig, ax = plt.subplots(2, 1, figsize=(20, 10))
sns.kdeplot(data=dfData, x='risk', label='risk', ax=ax[0])
ax[0].set_xlabel("Risk (mDKK)")
ax[0].set_ylabel("Density")
ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1).get_frame().set_linewidth(0.0)
# Limit x-axis to cover 99.99% of the data
ax[0].set_xlim(dfData['risk'].quantile(0.01), 0.99)
sns.lineplot(x='date', y='risk', hue='department', data=dfData, errorbar=None, ax=ax[1])
ax[1].set_xlabel("Date")
ax[1].set_ylabel("Risk")
ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/1_3_risk.png")
plt.savefig("./Results/Presentation/1_3_risk.svg")
upload(plt, 'Project-based Internship', 'figures/1_3_risk.png')

# Plot distribution of budget_costs, sales_estimate_costs, production_estimate_costs and final_estimate_costs
plt.figure(figsize=(20, 10))
sns.kdeplot(data=dfData, x='risk', label='risk')
plt.xlabel("Risk (mDKK)")
plt.ylabel("Density")
plt.xlim(dfData['risk'].quantile(0.01), dfData['risk'].quantile(0.99))
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=1).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/1_3_risk.png")
plt.savefig("./Results/Presentation/1_3_risk.svg")
upload(plt, 'Project-based Internship', 'figures/1_3_risk.png')


### Missing Data Analysis ###
# Calculate missing values
dfMissing = dfData.isna().sum().reset_index()
dfMissing.columns = ['column', 'missing']

# Calculate missing percentage
dfMissing['missing_pct'] = dfMissing['missing'] / dfData.shape[0]

# Sort by missing percentage
dfMissing.sort_values('missing_pct', ascending=False, inplace=True)

# Plot missing percentage above 0%
plt.figure(figsize=(20, 10))
sns.barplot(x=dfMissing[dfMissing['missing_pct'] > 0]['column'],
            y=dfMissing[dfMissing['missing_pct'] > 0]['missing_pct'])
plt.xticks(rotation=90)
plt.xlabel("Columns")
plt.ylabel("Missing Percentage")
plt.grid(alpha=0.5)
plt.rcParams['axes.axisbelow'] = True
plt.savefig("./Results/Figures/1_4_missing.png")
plt.savefig("./Results/Presentation/1_4_missing.svg")
upload(plt, 'Project-based Internship', 'figures/1_4_missing.png')

# Plot kde of labor_cost_share, material_cost_share and other_cost_share
plt.figure(figsize=(20, 10))
sns.kdeplot(data=dfData, x='labor_cost_share', label='labor cost share')
sns.kdeplot(data=dfData, x='material_cost_share', label='material cost share')
plt.xlabel("Share")
plt.ylabel("Density")
plt.xlim(0, 1)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.grid(alpha=0.35)
plt.savefig("./Results/Figures/1_5_cost_share.png")
plt.savefig("./Results/Presentation/1_5_cost_share.svg")
upload(plt, 'Project-based Internship', 'figures/1_5_cost_share.png')


# Create a figure with 4 subplots (1 column, 4 rows)
fig, axs = plt.subplots(4, 1, figsize=(35, 20))
# Plot 1: Risk vs Total Contribution
sns.scatterplot(ax=axs[0], data=dfData, x='risk', y='total_contribution', hue='department')
sns.regplot(ax=axs[0], data=dfData[dfData['department'] == '505'], x='risk', y='total_contribution', scatter=False, color=vColors[1])
sns.regplot(ax=axs[0], data=dfData[dfData['department'] == '515'], x='risk', y='total_contribution', scatter=False, color=vColors[2])
axs[0].set_xlabel("Risk (mDKK)")
axs[0].set_ylabel("Total Contribution (mDKK)")
axs[0].grid(alpha=0.35)
# Plot 2: Billable Rate vs Total Contribution
sns.scatterplot(ax=axs[1], data=dfData, x='billable_rate_dep', y='total_contribution', hue='department')
sns.regplot(ax=axs[1], data=dfData[dfData['department'] == '505'], x='billable_rate_dep', y='total_contribution', scatter=False, color=vColors[1])
sns.regplot(ax=axs[1], data=dfData[dfData['department'] == '515'], x='billable_rate_dep', y='total_contribution', scatter=False, color=vColors[2])
axs[1].set_xlabel("Billable Rate")
axs[1].set_ylabel("Total Contribution (mDKK)")
axs[1].grid(alpha=0.35)
# Plot 3: Project Duration vs Total Contribution
sns.scatterplot(ax=axs[2], data=dfData, x='total_days', y='total_contribution', hue='department')
sns.regplot(ax=axs[2], data=dfData[dfData['department'] == '505'], x='total_days', y='total_contribution', scatter=False, color=vColors[1])
sns.regplot(ax=axs[2], data=dfData[dfData['department'] == '515'], x='total_days', y='total_contribution', scatter=False, color=vColors[2])
axs[2].set_xlabel("Project Duration (days)")
axs[2].set_ylabel("Total Contribution (mDKK)")
axs[2].grid(alpha=0.35)
# Plot 4: Costs S-Curve Difference vs Total Contribution
sns.scatterplot(ax=axs[3], data=dfData, x='costs_scurve_diff', y='total_contribution', hue='department')
sns.regplot(ax=axs[3], data=dfData[dfData['department'] == '505'], x='costs_scurve_diff', y='total_contribution', scatter=False, color=vColors[1])
sns.regplot(ax=axs[3], data=dfData[dfData['department'] == '515'], x='costs_scurve_diff', y='total_contribution', scatter=False, color=vColors[2])
axs[3].set_xlabel("Costs S-Curve Difference (mDKK)")
axs[3].set_ylabel("Total Contribution (mDKK)")
axs[3].grid(alpha=0.35)
# Legend
axs[0].legend().remove()
axs[1].legend().remove()
axs[2].legend().remove()
axs[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2).get_frame().set_linewidth(0.0)
plt.savefig("./Results/Presentation/analysis_scatter.svg")


# Summary of Variables (mean, std, min, max, missing, % missing)
summary_data = dfData.select_dtypes(exclude=['datetime']).describe().transpose()
summary_data_date = dfData.select_dtypes(include=['datetime']).describe().transpose()
# Keep index, min, max, mean, std.
formatted_df_eda_1 = summary_data[['mean', 'std', 'min', 'max']]
formatted_df_eda_1_date = summary_data_date[['min', 'max']]
# Rename first = min and last = max
formatted_df_eda_1_date = formatted_df_eda_1_date.rename(columns={'first': 'min', 'last': 'max'})
# Format as dd-mm-yyyy
formatted_df_eda_1_date = formatted_df_eda_1_date.applymap(lambda x: x.strftime('%d-%m-%Y'))
formatted_df_eda_1_date.insert(0, 'mean', np.nan)
formatted_df_eda_1_date.insert(1, 'std', np.nan)
# Count number of missing values for each variable
missing_values = dfData.isnull().sum().to_frame()
# Rename column
missing_values = missing_values.rename(columns={0: 'missing'})
# Percentage of missing values
missing_values['% missing'] = missing_values['missing'] / len(dfData) * 100
# formatted_df_eda_1 = formatted_df_eda_1.select_dtypes(include=[np.number]).map('{:,.2f}'.format)
# Add mean and std to formatted_df_eda_1_date (set to NA)
formatted_df_eda_1_date['mean'] = np.nan
formatted_df_eda_1_date['std'] = np.nan
# Join formatted_df_eda_1 and formatted_df_eda_1_date. Date variables are first.
# AttributeError: 'DataFrame' object has no attribute 'append'
formatted_df_eda_1 = pd.concat([formatted_df_eda_1_date, formatted_df_eda_1], axis=0)
# Join missing
formatted_df_eda_1 = formatted_df_eda_1.join(missing_values)

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
upload(eda_1, 'Project-based Internship', 'tables/2_eda_1.tex')

plt.close('all')
