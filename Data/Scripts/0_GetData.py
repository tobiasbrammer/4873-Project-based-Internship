import pandas as pd
import numpy as np
import os
import datetime
import sqlalchemy as sa
import pyodbc
import urllib
from sqlalchemy import create_engine, event
from sqlalchemy.engine.url import URL
import pyarrow as pa
import pyarrow.parquet as pq

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

# Convert specified columns to categorical data type
factor_cols = ['month', 'year', 'job_posting_group', 'department', 'status', 'responsible']
dfData[factor_cols] = dfData[factor_cols].astype('category')

# Save DataFrame to file
dfData.to_csv("dfData.csv", index=False)
pq.write_table(pa.table(dfData), "dfData.parquet")

# End timing and print duration
end_time = datetime.datetime.now()
print(f"Time taken: {end_time - start_time}")
