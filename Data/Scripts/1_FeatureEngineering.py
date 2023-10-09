import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
import re
from plot_config import *

# Set working directory
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(sDir)

# Read data
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")

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

# Plot distribution of budget_costs, sales_estimate_costs, production_estimate_costs and final_estimate_costs
plt.figure(figsize=(10, 5))
sns.kdeplot(data=dfData, x='budget_costs', label='budget costs')
sns.kdeplot(data=dfData, x='sales_estimate_costs', label='sales estimate costs')
sns.kdeplot(data=dfData, x='production_estimate_costs', label='production estimate costs')
sns.kdeplot(data=dfData, x='final_estimate_costs', label='final estimate costs')
plt.xlabel("Costs (mDKK)")
plt.ylabel("Density")
plt.xlim(dfData['budget_costs'].quantile(0.00000000001), 10)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.25),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.show()
plt.savefig("./Results/Figures/1_0_costs.png")
plt.savefig("./Results/Presentation/1_0_costs.svg")

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
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
plt.tight_layout()
plt.grid(alpha=0.35)
plt.annotate('Source: ELCON A/S',
             xy=(1.0, -0.25),
             color='grey',
             xycoords='axes fraction',
             ha='right',
             va="center",
             fontsize=10)
plt.show()
plt.savefig("./Results/Figures/1_1_revenue.png")
plt.savefig("./Results/Presentation/1_1_revenue.svg")


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

# Show jobs with highest risk
dfData.sort_values('risk', ascending=False).head(15)

# Plot sum of risk by date for each department
plt.figure(figsize=(10, 5))
sns.lineplot(x='date', y='risk', hue='department', data=dfData)
plt.xlabel("Date")
plt.ylabel("Risk")
# legend below x-axis label
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
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
plt.savefig("./Results/Figures/1_2_risk.png")
plt.savefig("./Results/Presentation/1_2_risk.svg")

# Select random job and plot risk
job_no = dfData['job_no'].drop_duplicates().sample().values[0]
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

# Calculate total costs at the end of the job
dfData['total_costs'] = dfData.groupby('job_no')['costs_cumsum'].transform('last')
dfData['total_contribution'] = dfData.groupby('job_no')['contribution_cumsum'].transform('last')
dfData['total_margin'] = dfData['total_contribution'] / dfData['total_costs']

# Calculate contribution margin as contribution_cumsum / costs_cumsum
dfData['contribution_margin'] = dfData['contribution_cumsum'] / dfData['costs_cumsum']

# Calculate share of labor cost, material cost and other cost cumsum
dfData['labor_cost_share'] = dfData['costs_of_labor_cumsum'] / dfData['costs_cumsum']
dfData['material_cost_share'] = dfData['costs_of_materials_cumsum'] / dfData['costs_cumsum']
dfData['other_cost_share'] = dfData['other_costs_cumsum'] / dfData['costs_cumsum']

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

# Step 5: Find and plot the most frequent terms
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
plt.show()
plt.savefig("./Results/Figures/1_3_description.png")
plt.savefig("./Results/Presentation/1_3_description.svg")

# Step 6: Append the Document-Term Matrix to the original DataFrame
dfDesc.reset_index(drop=True, inplace=True)
df_matrix.reset_index(drop=True, inplace=True)
processed_data = pd.concat([dfDesc[['job_no']], df_matrix], axis=1)

# Left join with the original DataFrame
dfData = pd.merge(dfData, processed_data, on="job_no", how="left")

# Remove description from dfData
dfData.drop(columns=['description'], inplace=True)

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
plt.show()
plt.savefig("./Results/Figures/1_4_missing.png")
plt.savefig("./Results/Presentation/1_4_missing.svg")

### Split test and train ###
# Sample 80% of the jobs for training
lJobNoTrain = dfData['job_no'].drop_duplicates().sample(frac=0.8)
dfData['train'] = dfData['job_no'].isin(lJobNoTrain).astype(int)

# Save dfData to parquet file
dfData.to_parquet("./dfData.parquet")
