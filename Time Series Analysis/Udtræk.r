library(DBI)
library(RMySQL)
install.packages("RMySQL")
# Connect to the MySQL database: con
con <- dbConnect(odbc::odbc(), 
                 .connection_string = 'driver={SQL Server};server=[SARDUSQLBI01];database=[NRGIDW_Extract];trusted_connection=true')


# # Get table names
tables <- dbListTables(con)

# Display structure of tables
str(tables)