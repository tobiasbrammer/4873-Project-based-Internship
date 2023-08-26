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


dfData <- data.frame(dbFetch(dbSendQuery(con,sQuery)))

# Set date as date
dfData$date <- as.Date(dfData$date)
dfData$end_date <- as.Date(dfData$end_date)

# If end_date is NA set to date
dfData$end_date[is.na(dfData$end_date)] <- dfData$date[is.na(dfData$end_date)]

# If end_date is = 1753-01-01, set to date
dfData$end_date[dfData$end_date == as.Date("1753-01-01")] <- dfData$date[dfData$end_date == as.Date("1753-01-01")]

colMil <- names(dfData)[sapply(dfData, is.numeric)]
colMil <- colMil[!grepl("_share",colMil)]

dfData[,colMil] <- dfData[,colMil]/1000000

# Specify
colFact <- c('month','year','job_posting_group','department','status','responsible')

# Set factor
dfData[,colFact] <- lapply(dfData[,colFact], factor)

# Save dfData to file
write.csv(dfData,"dfData.csv")
write_parquet(dfData, "dfData.parquet")

# Close connection
dbDisconnect(con)

# End timer
Sys.time() - start_time