import pandas as pd
import numpy as np
import os
import sys
import datetime
import sqlalchemy as sa
import pyodbc
import urllib
from sqlalchemy import create_engine, event
from sqlalchemy.engine.url import URL
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
import re
from plot_config import *

# Start timing
start_time = datetime.datetime.now()

# Set the directory
directory = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(directory)

# If time is 05:00 - 05:15, then wait until 05:15
if datetime.datetime.now().hour == 5 and datetime.datetime.now().minute < 15:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 15:
        pass
# If time is 8:00 - 8:15, then wait until 8:15
elif datetime.datetime.now().hour == 8 and datetime.datetime.now().minute < 15:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 15:
        pass
# If time is 11:30 - 11:45, then wait until 11:45
elif datetime.datetime.now().hour == 11 and datetime.datetime.now().minute < 45:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 45:
        pass
# If time is 14:00 - 14:15, then wait until 14:15
elif datetime.datetime.now().hour == 14 and datetime.datetime.now().minute < 15:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 15:
        pass
# If time is 17:00 - 17:15, then wait until 17:15
elif datetime.datetime.now().hour == 17 and datetime.datetime.now().minute < 15:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 15:
        pass
# If time is 21:00 - 21:15, then wait until 21:15
elif datetime.datetime.now().hour == 21 and datetime.datetime.now().minute < 15:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 15:
        pass
# If time is 02:00 - 02:15, then wait until 02:15
elif datetime.datetime.now().hour == 2 and datetime.datetime.now().minute < 15:
    print("Waiting 15 minutes...")
    while datetime.datetime.now().minute < 15:
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
with open(".SQL/Data_v4.sql", "r") as file:
    sQuery = file.read()

dfData = pd.DataFrame()

# Test connection
with engine.begin() as conn:
    dfData = pd.read_sql_query(sa.text(sQuery), conn)

# Convert date columns to datetime format
dfData['date'] = pd.to_datetime(dfData['date'], format='%d-%m-%Y')
dfData['end_date'] = pd.to_datetime(dfData['end_date'], format='%d-%m-%Y')

# Fill NA values in end_date with date
dfData['end_date'].fillna(dfData['date'], inplace=True)

# Replace end_date with date if end_date is 1753-01-01
mask = dfData['end_date'] == '1753-01-01'
dfData.loc[mask, 'end_date'] = dfData.loc[mask, 'date']

# Divide numeric columns by 1,000,000
numeric_cols = dfData.select_dtypes(include=['number']).columns
numeric_cols = [col for col in numeric_cols if "_share" not in col and "_qty" not in col]
dfData[numeric_cols] = dfData[numeric_cols] / 1000000

# If zip code is more than 4 digits, set to NA
dfData.loc[dfData['zip_code'].astype(str).str.len() > 4, 'zip_code'] = np.nan

# Convert specified columns to categorical data type
factor_cols = ['month', 'year', 'job_posting_group', 'department', 'status', 'responsible']
dfData[factor_cols] = dfData[factor_cols].astype('category')

# Order by date
dfData.sort_values('date', inplace=True)

### Explore numeric variables ###
# Identify numeric columns based on conditions
colNum = [col for col in dfData.select_dtypes(include=[np.number]).columns if
          not any(sub in col for sub in ["_share", "_rate", "_ratio", "_margin", "_cumsum"])]
colCumSum = [col for col in colNum if "budget_" not in col]

# Calculate cumulative sum for each variable in colCumSum grouped by 'job_no'
for col in colCumSum:
    dfData[f'{col}_cumsum'] = dfData.groupby('job_no')[col].cumsum()

# Calculate days until end of job
dfData['days_until_end'] = (dfData['end_date'] - dfData['date']).dt.days
dfData.loc[dfData['days_until_end'] < 0, 'days_until_end'] = 0

# Calculate share of budget costs and budget revenue
dfData['budget_costs_share'] = dfData['revenue'] / dfData['budget_revenue']
dfData['budget_revenue_share'] = dfData['costs'] / dfData['budget_costs']

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

k = 6  # Coefficient for S-curve
a = 2  # Exponent for S-curve
dfData['scurve'] = (1 / (1 + np.exp(-k * (dfData['progress'] - 0.5)))) ** a
dfData['revenue_scurve'] = dfData['scurve'] * dfData['budget_revenue']
dfData['costs_scurve'] = dfData['scurve'] * dfData['budget_costs']
dfData['revenue_scurve_diff'] = dfData['revenue_scurve'] - dfData['revenue_cumsum']
dfData['costs_scurve_diff'] = dfData['costs_scurve'] - dfData['costs_cumsum']
dfData['contribution_scurve'] = dfData['scurve'] * (dfData['budget_revenue'] - dfData['budget_costs'])
dfData['contribution_scurve_diff'] = dfData['contribution_scurve'] - dfData['contribution_cumsum']

# Calculate risks and other variables
def calculate_risk(group):
    if group['contribution_scurve_diff'].isna().any() or group['contribution_cumsum'].isna().any():
        group['risk'] = np.nan
    else:
        X = group[
            ['revenue_scurve_diff', 'costs_scurve_diff', 'billable_rate_dep']]
        y = group['contribution_scurve_diff']
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        group['risk'] = residuals*group['budget_costs']
    return group

dfData = dfData.groupby('job_no', group_keys=False).apply(calculate_risk)


# Calculate total costs at the end of the job
dfData['total_costs'] = dfData.groupby('job_no')['costs_cumsum'].transform('last')
dfData['total_contribution'] = dfData.groupby('job_no')['contribution_cumsum'].transform('last')
dfData['total_margin'] = dfData['total_contribution'] / dfData['total_costs']

# Calculate contribution margin as contribution_cumsum / costs_cumsum
dfData['contribution_margin'] = dfData['contribution_cumsum'] / dfData['costs_cumsum']

# Calculate share of labor cost, material cost and other cost cumsum
dfData['labor_cost_share'] = dfData['costs_of_labor_cumsum'] / dfData['costs_cumsum']
dfData['material_cost_share'] = (dfData['costs_of_materials_cumsum']+dfData['other_costs_cumsum']) / dfData['costs_cumsum']

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

### Text Processing ###
# Step 1: Filter out the latest description for each job_no
dfDesc = dfData.sort_values('date').groupby('job_no').last().reset_index()
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

# Step 3 and 4: Convert to Document-Term Matrix and remove sparse terms
vectorizer = CountVectorizer(min_df=0.01, max_df=0.15)
X = vectorizer.fit_transform(dfDesc['description'])
df_matrix = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Step 5: Append the Document-Term Matrix to the original DataFrame
dfDesc.reset_index(drop=True, inplace=True)
df_matrix.reset_index(drop=True, inplace=True)
processed_data = pd.concat([dfDesc[['job_no']], df_matrix], axis=1)

term_frequencies = df_matrix.sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
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
plt.savefig("./Results/Figures/1_3_description.png")
plt.savefig("./Results/Presentation/1_3_description.svg")
plt.show()
plt.draw()

# Left join with the original DataFrame
dfData = pd.merge(dfData, processed_data, on="job_no", how="left")

# Remove description from dfData
dfData.drop(columns=['description'], inplace=True)

### Split test and train ###
# Sample 80% of the jobs for training
lJobNoTrain = dfData['job_no'].drop_duplicates().sample(frac=0.8)
dfData['train'] = dfData['job_no'].isin(lJobNoTrain).astype(int)

# Save DataFrame to file
dfData.to_csv("dfData.csv", index=False)
pq.write_table(pa.table(dfData), "dfData.parquet")

# End timing and print duration
end_time = datetime.datetime.now()
print(f"Time taken: {end_time - start_time}")
