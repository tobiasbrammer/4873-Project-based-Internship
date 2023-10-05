import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
import re

# Set working directory
dir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"

# Read data
dfData = pd.read_parquet(f"{dir}/dfData.parquet")

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
    dfData[f'{estimate_type}_estimate_contribution_change'] = dfData.groupby('job_no')[f'{estimate_type}_estimate_contribution'].diff().fillna(0)

# S-curve calculations
dfData['days_since_start'] = (dfData['date'] - dfData.groupby('job_no')['date'].transform('min')).dt.days
dfData['total_days'] = (dfData.groupby('job_no')['end_date'].transform('max') - dfData.groupby('job_no')['date'].transform('min')).dt.days
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
        X = group[['revenue_scurve_diff', 'costs_scurve_diff', 'billable_rate_dep', 'illness_rate_dep', 'internal_rate_dep']]
        y = group['contribution_scurve_diff']
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        group['risk'] = np.exp(residuals)
    return group

dfData = dfData.groupby('job_no').apply(calculate_risk)

# Calculate total costs at the end of the job
dfData['total_costs'] = dfData.groupby('job_no')['costs_cumsum'].transform('last')
dfData['total_contribution'] = dfData.groupby('job_no')['contribution_cumsum'].transform('last')
dfData['total_margin'] = dfData['total_contribution'] / dfData['total_costs']

# Calculate contribution margin as contribution_cumsum / costs_cumsum
dfData['contribution_margin'] = dfData['contribution_cumsum'] / dfData['costs_cumsum']

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
term_frequencies.head(10).plot(kind='bar')
plt.title("Top 10 Most Frequent Words")
plt.xlabel("Words")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
plt.savefig("./Results/Figures/0_description.pdf")
plt.savefig("./Results/Presentation/0_description.svg")

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

### Split test and train ###
# Sample 80% of the jobs for training
lJobNoTrain = dfData['job_no'].drop_duplicates().sample(frac=0.8)
dfData['train'] = dfData['job_no'].isin(lJobNoTrain).astype(int)

# Quality Assurance and Data Cleaning
# Filter the last date for each job_no
dfDataQA = dfData[dfData.groupby('job_no')['date'].transform('max') == dfData['date']]

# Sort by budget_costs and take the top 5 largest jobs
dfDataQA = dfDataQA.sort_values(by='budget_costs', ascending=False).head(5)

# Select relevant columns
dfDataQA = dfDataQA[['job_no', 'date', 'costs_cumsum', 'budget_costs', 'sales_estimate_costs', 'production_estimate_costs', 'final_estimate_costs']]


