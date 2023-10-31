import pandas as pd
import numpy as np
import os
import datetime
import sqlalchemy as sa
import urllib
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
import re
import PyDST

# Start timing
# start_time = datetime.datetime.now()

# Set the directory
# If operating system is macOS then
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

def upload(ax, project, path):
    bs = BytesIO()
    format = path.split('.')[-1]

    # Check if the file is a .tex file and handle it differently
    if format == 'tex':
        # Assuming the 'ax' parameter contains the LaTeX content
        content = ax
        format = 'tex'
    else:
        ax.savefig(bs, bbox_inches='tight', format=format)

    # token = os.DROPBOX
    token = subprocess.run("curl https://api.dropbox.com/oauth2/token -d grant_type=refresh_token -d refresh_token=eztXuoP098wAAAAAAAAAAV4Ef4mnx_QpRaiqNX-9ijTuBKnX9LATsIZDPxLQu9Nh -u a415dzggdnkro3n:00ocfqin8hlcorr", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.split('{"access_token": "')[1].split('", "token_type":')[0]
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    if format == 'tex':
        # Handle .tex files by directly uploading their content
        dbx.files_upload(content.encode(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)
    else:
        dbx.files_upload(bs.getvalue(), f'/Apps/Overleaf/{project}/{path}', mode=dropbox.files.WriteMode.overwrite)


from plot_config import *

# If time is 05:00 - 05:30, then wait until 05:30
if datetime.datetime.now().hour == 5 and datetime.datetime.now().minute < 30:
    print("Waiting until 5:30...")
    while datetime.datetime.now().minute < 30:
        pass
# If time is 8:00 - 8:15, then wait until 8:15
elif datetime.datetime.now().hour == 8 and datetime.datetime.now().minute < 30:
    print("Waiting until 8:30...")
    while datetime.datetime.now().minute < 30:
        pass
# If time is 11:30 - 11:45, then wait until 11:45
elif datetime.datetime.now().hour == 11 and 59 > datetime.datetime.now().minute >= 30:
    print("Waiting until 12:00...")
    while datetime.datetime.now().minute < 59:
        pass
# If time is 14:00 - 14:15, then wait until 14:15
elif datetime.datetime.now().hour == 14 and datetime.datetime.now().minute < 30:
    print("Waiting until 14:30...")
    while datetime.datetime.now().minute < 30:
        pass
# If time is 17:00 - 17:15, then wait until 17:15
elif datetime.datetime.now().hour == 17 and datetime.datetime.now().minute < 30:
    print("Waiting until 17:30...")
    while datetime.datetime.now().minute < 30:
        pass
# If time is 21:00 - 21:15, then wait until 21:15
elif datetime.datetime.now().hour == 21 and datetime.datetime.now().minute < 30:
    print("Waiting until 21:30...")
    while datetime.datetime.now().minute < 30:
        pass
# If time is 02:00 - 02:15, then wait until 02:15
elif datetime.datetime.now().hour == 2 and datetime.datetime.now().minute < 30:
    print("Waiting until 2:30...")
    while datetime.datetime.now().minute < 30:
        pass

# Connect to the database using SQLAlchemy
username = os.getenv("USERNAME")
server = "SARDUSQLBI01"
database = "NRGIDW_Extract"
params = urllib.parse.quote_plus("DRIVER={SQL Server};"
                                 "SERVER=" + server + ";"
                                                      "DATABASE=" + database + ";"
                                                                               "trusted_connection=yes")
engine = sa.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))

# Read SQL query from file
with open(".SQL/Data_v5.sql", "r") as file:
    sQuery = file.read()

print('Getting data from database...')

# Test connection
with engine.begin() as conn:
    dfData = pd.read_sql_query(sa.text(sQuery), conn)

print('Running feature engineering...')

# Convert date columns to datetime format
dfData['date'] = pd.to_datetime(dfData['date'], format='%d-%m-%Y')
dfData['end_date'] = pd.to_datetime(dfData['end_date'], format='%d-%m-%Y')

# Fill NA values in end_date with date
dfData['end_date'].fillna(dfData['date'], inplace=True)

# Replace end_date with date if end_date is 1753-01-01
mask = dfData['end_date'] == '1753-01-01'
dfData.loc[mask, 'end_date'] = dfData.loc[mask, 'date']

dfData.loc[dfData['zip'].astype(str).str.len() > 4, 'zip'] = np.nan
dfData.loc[dfData['customer_zip'].astype(str).str.len() > 4, 'customer_zip'] = np.nan

# If sales_estimate_revenue, sales_estimate_costs, production_estimate_revenue, production_estimate_costs, final_estimate_revenue or final_estimate_costs is 0, then set to budget_revenue or budget_costs
dfData.loc[dfData['sales_estimate_revenue'] == 0, 'sales_estimate_revenue'] = dfData.loc[
    dfData['sales_estimate_revenue'] == 0, 'budget_revenue']
dfData.loc[dfData['sales_estimate_costs'] == 0, 'sales_estimate_costs'] = dfData.loc[
    dfData['sales_estimate_costs'] == 0, 'budget_costs']
dfData.loc[dfData['production_estimate_revenue'] == 0, 'production_estimate_revenue'] = dfData.loc[
    dfData['production_estimate_revenue'] == 0, 'budget_revenue']
dfData.loc[dfData['production_estimate_costs'] == 0, 'production_estimate_costs'] = dfData.loc[
    dfData['production_estimate_costs'] == 0, 'budget_costs']
dfData.loc[dfData['final_estimate_revenue'] == 0, 'final_estimate_revenue'] = dfData.loc[
    dfData['final_estimate_revenue'] == 0, 'budget_revenue']
dfData.loc[dfData['final_estimate_costs'] == 0, 'final_estimate_costs'] = dfData.loc[
    dfData['final_estimate_costs'] == 0, 'budget_costs']

# Divide numeric columns by 1,000,000
numeric_cols = dfData.select_dtypes(include=['number']).columns
numeric_cols = [col for col in numeric_cols if "_share" not in col and "_qty" not in col and "_rate" not in col]
dfData[numeric_cols] = dfData[numeric_cols] / 1000000

# If zip code is more than 4 digits, set to NA
dfData.loc[dfData['zip'].astype(str).str.len() > 4, 'zip'] = np.nan
dfData.loc[dfData['customer_zip'].astype(str).str.len() > 4, 'customer_zip'] = np.nan

# Convert specified columns to categorical data type
factor_cols = ['month', 'year', 'job_posting_group', 'department', 'status', 'responsible']
dfData[factor_cols] = dfData[factor_cols].astype('category')

# Order by date
dfData.sort_values('date', inplace=True)

### Explore numeric variables ###
# Identify numeric columns based on conditions
colNum = [col for col in dfData.select_dtypes(include=[np.number]).columns if
          not any(sub in col for sub in ["_share", "_rate", "_ratio", "_margin", "_cumsum", "_estimate_"])]
colCumSum = [col for col in colNum if "budget_" not in col]

# Save colCumSum to ./.AUX/colCumSum.txt
with open('./.AUX/colCumSum.txt', 'w') as f:
    f.write('\n'.join(colCumSum))

# Calculate cumulative sum for each variable in colCumSum grouped by 'job_no'
for col in colCumSum:
    dfData[f'{col}_cumsum'] = dfData.groupby('job_no')[col].cumsum()

# Calculate days until end of job
dfData['days_until_end'] = (dfData['end_date'] - dfData['date']).dt.days
dfData.loc[dfData['days_until_end'] < 0, 'days_until_end'] = 0

# Calculate share of budget costs and budget revenue
dfData['budget_costs_share'] = (dfData['revenue'].replace(np.nan, 0) / dfData['budget_revenue']).replace([np.inf, -np.inf], 0)
dfData['budget_revenue_share'] = (dfData['costs'].replace(np.nan, 0) / dfData['budget_costs']).replace([np.inf, -np.inf], 0)

##### Feature engineering #####
# Calculate change in various estimates
for estimate_type in ['sales', 'production', 'final']:
    dfData[f'{estimate_type}_estimate_contribution_change'] = dfData.groupby('job_no')[
        f'{estimate_type}_estimate_contribution'].diff().fillna(0)

# S-curve calculations
dfData['days_since_start'] = (dfData['date'] - dfData.groupby('job_no')['date'].transform('min')).dt.days
dfData['total_days'] = (
        dfData.groupby('job_no')['end_date'].transform('max') - dfData.groupby('job_no')['date'].transform(
    'min')).dt.days
dfData['progress'] = dfData['days_since_start'] / dfData['total_days']
dfData['completion_rate'] = (dfData['costs_cumsum'] / dfData['production_estimate_costs']).replace([np.inf, -np.inf], 0)

# Calculate scurve basad on \Phi(x;\mu,\nu)=\left[1+\left(\frac{x\cdot(1-\mu)}{\mu\cdot(1-x)}\right)^{-\nu}\right]^{-1}
# where \mu is 0.5 and \nu is 1.5
dfData['scurve'] = 1 / (1 + (dfData['progress'] * (1 - 0.5) / (0.5 * (1 - dfData['progress']))) ** (-1.5))

dfData['revenue_scurve'] = dfData['scurve'] * dfData['budget_revenue']
dfData['costs_scurve'] = dfData['scurve'] * dfData['budget_costs']
dfData['revenue_scurve_diff'] = dfData['revenue_scurve'] - dfData['revenue_cumsum']
dfData['costs_scurve_diff'] = dfData['costs_scurve'] - dfData['costs_cumsum']
dfData['contribution_scurve'] = dfData['scurve'] * (dfData['budget_revenue'] - dfData['budget_costs'])
dfData['contribution_scurve_diff'] = dfData['contribution_scurve'] - dfData['contribution_cumsum']

# Calculate contribution margin as contribution_cumsum / costs_cumsum
dfData['contribution_margin'] = dfData['contribution_cumsum'] / dfData['costs_cumsum']

print('Getting data on WIP and overdue debtors...')

### Gather data from .AUX/Igv.xlsx ###
dfIgv = pd.read_excel(".AUX/Igv.xlsx", sheet_name="WIP")
# Rename 'adjusted_WIP' to 'adjusted_wip'
dfIgv.rename(columns={'adjusted_WIP': 'adjusted_wip'}, inplace=True)
# Divide adjusted_wip, adjusted_estimated_revenue by 1,000,000
dfIgv[['adjusted_wip', 'adjusted_estimated_revenue']] = dfIgv[['adjusted_wip', 'adjusted_estimated_revenue']] / 1000000
# Join on dfData by job_no and date
dfData = pd.merge(dfData, dfIgv, on=['job_no', 'date'], how='left')

# Calculate WIP as progress * budget_revenue - revenue_cumsum
dfData['wip'] = dfData['completion_rate'] * dfData['production_estimate_revenue'] - dfData['revenue_cumsum']
# If adjusted_wip is NA, then set to wip
dfData['adjusted_wip'] = (dfData['adjusted_wip']).replace([np.inf, -np.inf], np.nan).fillna(dfData['wip'])

# If adjusted_estimated_revenue is NA, then set to estimated_revenue
dfData['adjusted_estimated_revenue'] = dfData['adjusted_estimated_revenue'].fillna(dfData['estimated_revenue'])
# If adjusted_margin is NA, then set to contribution_margin
dfData['adjusted_margin'] = dfData['adjusted_margin'].replace([np.inf, -np.inf], np.nan).fillna(dfData['contribution_margin'])

# Read data from .AUX/Debitor.xlsx
dfDebitorer = pd.read_excel(".AUX/Debitorer.xlsx", sheet_name="overdue")
# Omit all columns except cvr, date and overdue
dfDebitorer = dfDebitorer[['date','cvr','overdue']]
dfDebitorer['date'] = pd.to_datetime(dfDebitorer['date'])
# Join on dfData by cvr and date
dfData = pd.merge(dfData, dfDebitorer, on=['cvr', 'date'], how='left')
# If overdue is NA, then set to 0
dfData['overdue'] = dfData['overdue'].fillna(0)
# Divide overdue by 1,000,000
dfData['overdue'] = dfData['overdue'] / 1000000

print('Running calculation of risk...')

# Calculate risks and other variables
def calculate_risk(group):
    if group['contribution_scurve_diff'].isna().any() or group['contribution_cumsum'].isna().any():
        group['risk'] = np.nan
    else:
        X = group[
            ['revenue_scurve_diff', 'costs_scurve_diff', 'billable_rate_dep']]
        y = group['contribution_scurve_diff']
        model = LinearRegression().fit(X.replace(np.nan, 0), y.replace(np.nan, 0))
        residuals = y - model.predict(X.replace(np.nan, 0))
        group['risk'] = residuals * group['production_estimate_costs']
    return group


# Apply calculation of risk to each job_no
dfData = dfData.groupby('job_no', group_keys=False).apply(calculate_risk)

# Determine the PACF and ACF of revenue, costs and contribution
for col in ['revenue', 'costs', 'contribution']:
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    plot_acf(dfData[col], ax=ax[0], lags=5, zero=False)
    plot_pacf(dfData[col], ax=ax[1], lags=5, zero=False)
    ax[0].set_ylim(-0.4)
    fig.suptitle(f'PACF and ACF of {col}')
    plt.tight_layout()
    plt.savefig(f"./Results/Figures/1_6_{col}_acf_pacf.png")
    plt.savefig(f"./Results/Presentation/1_6_{col}_acf_pacf.svg")
    upload(plt, 'Project-based Internship', f'figures/1_6_{col}_acf_pacf.png')

# Get 5 lagged values for revenue, costs and contribution for each job_no
for col in ['revenue', 'costs', 'contribution']:
    for i in range(1, 6):
        dfData[f'{col}_lag{i}'] = dfData.groupby('job_no', observed=True)[col].shift(i)

# For all lags replace NA with 0
for col in ['revenue', 'costs', 'contribution']:
    for i in range(1, 6):
        dfData[f'{col}_lag{i}'] = dfData[f'{col}_lag{i}'].fillna(0)

# Calculate total costs at the end of the job
dfData['total_costs'] = dfData.groupby('job_no')['costs_cumsum'].transform('last')
dfData['total_contribution'] = dfData.groupby('job_no')['contribution_cumsum'].transform('last')
dfData['total_margin'] = dfData['total_contribution'] / dfData['total_costs']

# Calculate share of labor cost, material cost and other cost cumsum

dfData['labor_cost_share'] = (dfData['costs_of_labor_cumsum'].replace(np.nan, 0) / dfData['costs_cumsum']).replace([np.inf, -np.inf], 0)
dfData['material_cost_share'] = ((dfData['costs_of_materials_cumsum'] + dfData['other_costs_cumsum']).replace(np.nan, 0) / dfData[
    'costs_cumsum']).replace([np.inf, -np.inf], 0)

# Omit labor_cost_cumsum, material_cost_cumsum and other_cost_cumsum
dfData.drop(columns=['costs_of_labor_cumsum', 'costs_of_materials_cumsum', 'other_costs_cumsum'], inplace=True)


# Function to set to NA if NaN, inf or -inf
def set_na(x):
    if np.isnan(x) or np.isinf(x) or x == -np.inf:
        return np.nan
    else:
        return x


# Set total_margin, contribution_margin and progress to NA if NaN, inf or -inf
dfData['total_margin'] = dfData['total_margin'].apply(set_na)
dfData['contribution_margin'] = dfData['contribution_margin'].apply(set_na)
dfData['progress'] = dfData['progress'].apply(set_na)

### Encode categorical variables ###
dfData['wip'] = (dfData['status'] == 'wip').astype(int)
dfData['dep_505'] = (dfData['department'] == '505').astype(int)
dfData['posting_group_projekt'] = (dfData['job_posting_group'] == 'PROJEKT').astype(int)

# Convert to categorical
for col in ['responsible', 'address', 'cvr', 'customer', 'job_no']:
    dfData[col] = dfData[col].astype('category')

print('Running text processing...')

### Text Processing ###
# Step 1: Filter out the latest description for each job_no
dfDesc = dfData.sort_values('date').groupby('job_no').last().reset_index()
# Replace ø with oe, æ with ae and å with aa
dfDesc = dfDesc[['job_no', 'description']]
dfDesc = dfDesc[dfDesc['description'] != ""]

# Step 2: Preprocess text
stemmer = DanishStemmer()
stop_words = stopwords.words('danish')


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text.strip()


dfDesc['description'] = dfDesc['description'].apply(preprocess)
dfDesc['description'] = dfDesc['description'].str.replace('ø', 'oe')
dfDesc['description'] = dfDesc['description'].str.replace('æ', 'ae')
dfDesc['description'] = dfDesc['description'].str.replace('å', 'aa')


# Step 3: Convert to Document-Term Matrix and remove sparse terms
vectorizer = CountVectorizer(min_df=0.02, max_df=0.15)
X = vectorizer.fit_transform(dfDesc['description'])
df_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Step 4: Append the Document-Term Matrix to the original DataFrame
dfDesc.reset_index(drop=True, inplace=True)
df_matrix.reset_index(drop=True, inplace=True)
processed_data = pd.concat([dfDesc[['job_no']], df_matrix], axis=1)

term_frequencies = df_matrix.sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(20, 10))
sns.barplot(x=term_frequencies.index, y=term_frequencies.values)
plt.xticks(rotation=90)
plt.xlabel("Terms")
plt.ylabel("Frequency")
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.35),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.savefig("./Results/Figures/1_7_description.png")
plt.savefig("./Results/Presentation/1_7_description.svg")
upload(plt, 'Project-based Internship', 'figures/1_7_description.png')


# Left join with the original DataFrame
dfData = pd.merge(dfData, processed_data, on="job_no", how="left")

# Remove description from dfData
dfData.drop(columns=['description'], inplace=True)

## Encode 'category' column ##
# Create dummy variables for 'category' column
# If 'category' is NA, then set to 'No_category'
dfData['category'] = dfData['category'].fillna('No_category')
# If 'category' is ' ', then set to 'No_category'
dfData.loc[dfData['category'] == ' ', 'category'] = 'No_category'
dfData.loc[dfData['category'] == '', 'category'] = 'No_category'
# Strip leading and trailing whitespace
dfData['category'] = dfData['category'].str.strip()
# Replace ' ' with '_'
dfData['category'] = dfData['category'].str.replace(' ', '_')
#
dfData = pd.concat([dfData, pd.get_dummies(dfData['category'])], axis=1)
dfData.drop(columns=['category'], inplace=True)

print('Getting data from DST...')

### Join DST data ###
kbyg11 = PyDST.get_data(table_id='KBYG11',
                        variables={'BRANCHE07': '43201',
                                   'INDIKATOR': 'VIAK',
                                   'BEDØMMELSE': 'NET',
                                   'FORLØB': 'FAK',
                                   'TID': '*'})
kbyg11 = PyDST.utils.to_dataframe(kbyg11)
# Extract date from 'TID' column. (YYYY'M'MM) -> (YYYY-MM)
kbyg11['date'] = kbyg11['TID'].str[:4] + '-' + kbyg11['TID'].str[5:7]
# Convert date column to datetime format
kbyg11['date'] = pd.to_datetime(kbyg11['date'], format='%Y-%m')
# Convert to dd-mm-YYYY format
kbyg11['date'] = kbyg11['date'].dt.strftime('%d-%m-%Y')
kbyg11['date'] = pd.to_datetime(kbyg11['date'], format='%d-%m-%Y')
# Rename INDHOLD to 'kbyg11'
kbyg11.rename(columns={'INDHOLD': 'kbyg11'}, inplace=True)
# Divide 'kbyg11' by 100
kbyg11['kbyg11'] = kbyg11['kbyg11'] / 100
# Omit all columns except 'date' and 'kbyg11'
kbyg11 = kbyg11[['date', 'kbyg11']]
# Convert 'kbyg11' to numeric
kbyg11['kbyg11'] = pd.to_numeric(kbyg11['kbyg11'], errors='coerce')

## KBYG22
kbyg22 = PyDST.get_data(table_id='KBYG22',
                        variables={'BRANCHE07': '43201',
                                   'BEDØMMELSE': 'NET',
                                   'TID': '*'})

kbyg22 = PyDST.utils.to_dataframe(kbyg22)
# Extract date from 'TID' column. (YYYY'M'MM) -> (YYYY-MM)
kbyg22['date'] = kbyg22['TID'].str[:4] + '-' + kbyg22['TID'].str[5:7]
# Convert date column to datetime format
kbyg22['date'] = pd.to_datetime(kbyg22['date'], format='%Y-%m')
# Convert to dd-mm-YYYY format
kbyg22['date'] = kbyg22['date'].dt.strftime('%d-%m-%Y')
kbyg22['date'] = pd.to_datetime(kbyg22['date'], format='%d-%m-%Y')
# Rename INDHOLD to 'kbyg22'
kbyg22.rename(columns={'INDHOLD': 'kbyg22'}, inplace=True)
kbyg22['kbyg22'] = kbyg22['kbyg22'] / 100
# Omit all columns except 'date' and 'kbyg22'
kbyg22 = kbyg22[['date', 'kbyg22']]
# Convert 'kbyg22' to numeric
kbyg22['kbyg22'] = pd.to_numeric(kbyg22['kbyg22'], errors='coerce')

kbyg33 = PyDST.get_data(table_id='KBYG33',
                        variables={'BRANCHE07': '43201',
                                   'TYPE': '*',
                                   'TID': '*'})
kbyg33 = PyDST.utils.to_dataframe(kbyg33)
# Extract date from 'TID' column. (YYYY'M'MM) -> (YYYY-MM)
kbyg33['date'] = kbyg33['TID'].str[:4] + '-' + kbyg33['TID'].str[5:7]
# Convert date column to datetime format
kbyg33['date'] = pd.to_datetime(kbyg33['date'], format='%Y-%m')
# Convert to dd-mm-YYYY format
kbyg33['date'] = kbyg33['date'].dt.strftime('%d-%m-%Y')
kbyg33['date'] = pd.to_datetime(kbyg33['date'], format='%d-%m-%Y')
# Rename INDHOLD to 'kbyg33'
kbyg33.rename(columns={'INDHOLD': 'kbyg33'}, inplace=True)
# Divide 'kbyg33' by 100
kbyg33['kbyg33'] = kbyg33['kbyg33'] / 100
# Omit all columns except 'date', 'TYPE' and 'kbyg33'
kbyg33 = kbyg33[['date', 'TYPE', 'kbyg33']]
# Pivot by 'TYPE'
kbyg33 = kbyg33.pivot(index='date', columns='TYPE', values='kbyg33').reset_index()
# rename columns lower case, prefix 'kbyg33_' and replace ' ' with '_'
kbyg33.columns = kbyg33.columns.str.lower().str.replace(' ', '_')
# Remove trailing _
kbyg33.columns = kbyg33.columns.str.rstrip('_')
# Prefix 'kbyg33_' to all columns except 'date'
kbyg33.columns = ['date'] + ['kbyg33_' + col for col in kbyg33.columns if col != 'date']

# KBYG44
kbyg44 = PyDST.get_data(table_id='KBYG44',
                        variables={'INDIKATOR': '*',
                                   'SÆSON': 'SÆSON',
                                   'TID': '*'})
kbyg44 = PyDST.utils.to_dataframe(kbyg44)
# Extract date from 'TID' column. (YYYY'M'MM) -> (YYYY-MM)
kbyg44['date'] = kbyg44['TID'].str[:4] + '-' + kbyg44['TID'].str[5:7]
# Convert date column to datetime format
kbyg44['date'] = pd.to_datetime(kbyg44['date'], format='%Y-%m')
# Convert to dd-mm-YYYY format
kbyg44['date'] = kbyg44['date'].dt.strftime('%d-%m-%Y')
kbyg44['date'] = pd.to_datetime(kbyg44['date'], format='%d-%m-%Y')
# Rename INDHOLD to 'kbyg44'
kbyg44.rename(columns={'INDHOLD': 'kbyg44'}, inplace=True)
# Divide 'kbyg44' by 100
kbyg44['kbyg44'] = kbyg44['kbyg44'] / 100
# Omit all columns except 'date', 'INDIKATOR' and 'kbyg44'
kbyg44 = kbyg44[['date', 'INDIKATOR', 'kbyg44']]
# Pivot by 'INDIKATOR'
kbyg44 = kbyg44.pivot(index='date', columns='INDIKATOR', values='kbyg44').reset_index()
# rename columns lower case, prefix 'kbyg44_' and replace ' ' with '_'
kbyg44.columns = kbyg44.columns.str.lower().str.replace(' ', '_')
kbyg44.columns = kbyg44.columns.str.rstrip('_')
# Replace ,_ by _
kbyg44.columns = kbyg44.columns.str.replace(',_', '_')
# Prefix 'kbyg44_' to all columns except 'date'
kbyg44.columns = ['date'] + ['kbyg44_' + col for col in kbyg44.columns if col != 'date']

# Join kbyg11 and kbyg22 on date
dst_df = pd.merge(kbyg11, kbyg22, on='date', how='left')
# Join dst_df and kbyg33 on date
dst_df = pd.merge(dst_df, kbyg33, on='date', how='left')
# Join dst_df and kbyg44 on date
dst_df = pd.merge(dst_df, kbyg44, on='date', how='left')

del kbyg11, kbyg22, kbyg33, kbyg44

# Rename kbyg44_construction_industry_total_employment_expectations to kbyg44_employment_expectations
dst_df.rename(columns={'kbyg44_construction_industry_total_employment_expectations': 'kbyg44_employment_expectations',
                       'kbyg44_confidence_indicator_total': 'kbyg44_confidence_indicator',
                       'kbyg33_shortage_of_material_and/or_equipment': 'kbyg33_shortage_of_materials'},
                inplace=True)


# Join dst_df on dfData by date
dfData = pd.merge(dfData, dst_df, on='date', how='left')


# Plot kbyg11, kbyg22, kbyg33_no_limitation and kbyg44_confidence_indicator_total by date
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
ax[0, 0].plot(dst_df['date'], dst_df['kbyg11'])
ax[0, 0].set_title('Change in Industry Revenue')
ax[0, 1].plot(dst_df['date'], dst_df['kbyg22'])
ax[0, 1].set_title('Assessment of Order Backlog')
ax[1, 0].plot(dst_df['date'], dst_df['kbyg33_no_limitations'])
ax[1, 0].set_title('Share with No Production Limitations')
ax[1, 1].plot(dst_df['date'], dst_df['kbyg44_confidence_indicator'])
ax[1, 1].set_title('Confidence indicator')
fig.suptitle('DST data')
plt.tight_layout()
plt.savefig("./Results/Figures/1_8_dst_data.png")
plt.savefig("./Results/Presentation/1_8_dst_data.svg")
upload(plt, 'Project-based Internship', 'figures/1_8_dst_data.png')

del dst_df

# Make column with concatenated date and job_no
dfData['id'] = dfData['date'].astype(str) + '_' + dfData['job_no'].astype(str)

# Omit observations before 2015-01-01
dfData = dfData[dfData['date'] >= '2015-01-01']

# Omit duplicate of 'job_no' and 'date'
dfData.drop_duplicates(subset=['id'], inplace=True)

# Format all column names to lower case
dfData.columns = dfData.columns.str.lower()

### Split test and train ###
# Sample 80% of the jobs for training. This allows us to effectively simulate training on finished jobs, and
# predicting on ongoing jobs.
lJobNoTrain = dfData['job_no'].drop_duplicates().sample(frac=0.8)
dfData['train'] = dfData['job_no'].isin(lJobNoTrain).astype(int)

# We can also use another method of split, where each job is split into a training and test set. This allows us to
# also simulate training ongoing jobs as they progress.
# We group by job_no and split the data into a training and test set for each job_no
dfData['train_TS'] = dfData.groupby('job_no')['job_no'].transform(lambda x: np.random.choice([0, 1], size=len(x), p=[.1, .9]))

# Import libraries
from sklearn.cluster import KMeans
# Run the process for five different numbers of clusters.
lCluster = [2, 4, 6, 8, 10, 12, 14]
# Assign cluster to each job
for nCluster in lCluster:
    # Create KMeans object
    kmeans = KMeans(n_clusters=nCluster, random_state=0, n_init='auto')
    # Fit the model
    kmeans.fit(dfData[dfData.select_dtypes(include=[np.number]).columns].replace([np.inf, -np.inf], np.nan).replace(np.nan, 0))
    # Predict the cluster for each observation
    dfData[f'cluster_{nCluster}'] = kmeans.predict(dfData[dfData.select_dtypes(include=[np.number]).columns].replace([np.inf, -np.inf], np.nan).replace(np.nan, 0))

# Save DataFrame to file
dfData.to_csv("dfData.csv", index=False)
pq.write_table(pa.table(dfData), "dfData.parquet")
pq.write_table(pa.table(dfData), "dfData_org.parquet")

# Close connection to database
engine.dispose()

plt.close('all')