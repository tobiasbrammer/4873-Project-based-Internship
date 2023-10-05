import pyodbc
import pandas as pd
import os
import datetime
from pyarrow import parquet as pq
import pyarrow as pa

# Start timing
start_time = datetime.datetime.now()

# Set the directory
directory = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(directory)

# Connect to the database using pyodbc
username = os.getenv("USERNAME")
connection_string = f'DRIVER={{SQL Server}}; DATABASE=NRGIDW_Extract; SERVER=SARDUSQLBI01; UID=NRGI\\{username}; Trusted_Connection=True'
con = pyodbc.connect(connection_string)

# Read SQL query from file
with open(".SQL/Data_v4.sql", "r") as file:
    sQuery = file.read()

# Fetch data into DataFrame
dfData = pd.read_sql(sQuery, con)

# Close the database connection
con.close()

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

# Convert specified columns to categorical data type
factor_cols = ['month', 'year', 'job_posting_group', 'department', 'status', 'responsible']
dfData[factor_cols] = dfData[factor_cols].astype('category')

# Save DataFrame to file
dfData.to_csv("dfData.csv", index=False)
pq.write_table(pa.table(dfData), "dfData.parquet")

# End timing and print duration
end_time = datetime.datetime.now()
print(f"Time taken: {end_time - start_time}")
