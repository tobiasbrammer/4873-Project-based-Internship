library(DBI)
library(RMySQL)
library(odbc)
library(lubridate)
library(readr)
library(arrow)

rm(list = ls())

# This script is used to extract data from the NRGIDW_Extract database and save it as a feather file.

# Set start time
start_time <- Sys.time()

# Get directory of file
dir <- "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
setwd(dir)

# Connect to database
con <- dbConnect(odbc::odbc(),
                 Driver = "SQL Server",
                 Database = 'NRGIDW_Extract',
                 Server = 'SARDUSQLBI01',
                 user = paste0("NRGI","\"",Sys.getenv("USERNAME")),
                 Trusted_Connection = "True")


sQuery <- read_file("C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data/.SQL/Data_v0.sql")

query <- dbSendQuery(con,sQuery)
dfData <- dbFetch(query)

# Set date as date
dfData$date <- as.Date(dfData$date)
dfData$end_date <- as.Date(dfData$end_date)

# If end_date is NA set to date
dfData$end_date[is.na(dfData$end_date)] <- dfData$date[is.na(dfData$end_date)]

# If end_date is = 1753-01-01, set to date
dfData$end_date[dfData$end_date == as.Date("1753-01-01")] <- dfData$date[dfData$end_date == as.Date("1753-01-01")]

# Format date to dd-MM-yyyy
dfData$date <- as.Date(format(dfData$date, "%d-%m-%Y"))
dfData$end_date <- as.Date(format(dfData$end_date, "%d-%m-%Y"))

colNum <- c('revenue','costs','costs_of_labor','costs_of_materials','other_costs','contribution','estimated_revenue','estimated_contribution')
colNum <- names(dfData)[sapply(dfData, is.numeric)]
dfData[,colNum] <- dfData[,colNum]/1000000

# Save dfData to file
write.csv(dfData,"dfData.csv")
write_parquet(dfData, "dfData.parquet")

dbClearResult(query)

# Close connection
dbDisconnect(con)

# End timer
Sys.time() - start_time

