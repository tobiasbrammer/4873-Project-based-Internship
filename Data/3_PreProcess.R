library(SmartEDA)
library(ExPanDaR)
library(arrow)
library(tibble)
library(ggplot2)
library(ggthemes)
library(ggbreak)
library(svglite)
library(dplyr)
library(tidyr)
library(texreg)
library(knitr)
library(kableExtra)
library(beepr)
library(patchwork)
library(caret)
library(pls)

# Source GetData
source('2_EDA.r')
rm(list=ls()[!grepl("dfData",ls())])
invisible(source('theme_elcon.R'))
rm(list=ls()[!grepl("vColor|theme_elcon|dfData",ls())])

# Describe variables in dfData and data type of each variable.
str(dfData)

### Split data into wip and finished jobs ###
dfDataFinished <- dfData %>% filter(wip == 0)
dfDataWIP <- dfData %>% filter(wip == 1)

# No of unique finished jobs and WIP jobs
nFinished <- length(unique(dfDataFinished$job_no))
nWIP <- length(unique(dfDataWIP$job_no))
paste0("The finished jobs account for ",round(nFinished/(nFinished+nWIP)*100,2),"% of the total number of jobs.")

# Calculate the average number of observations per finished job
paste0("The average number of observations per finished job is ",
       round(
             nrow(dfDataFinished)/nFinished,2),
       ".")
# Max number of observations per finished job
paste0("The max number of observations per finished job is ",
       max(dfDataFinished %>%
             group_by(job_no) %>%
             summarise(n = n()) %>%
             pull(n)),
       ".")
paste0("The min number of observations per finished job is ",
       min(dfDataFinished %>%
             group_by(job_no) %>%
             summarise(n = n()) %>%
             pull(n)),
       ".")

# Fraction of finished jobs with less than 4 observations
paste0("The fraction of finished jobs with less than 4 observations is ",
       round(sum(dfDataFinished %>%
                   group_by(job_no) %>%
                   summarise(n = n()) %>%
                   pull(n) < 4)/nFinished*100,2),
       "%.")

# Omit finished jobs with less than 4 observations
dfDataFinished <- dfDataFinished %>%
  group_by(job_no) %>%
  filter(n() >= 4) %>%
  ungroup()

### Split the finished jobs into train and test ###
dfDataFinishedTrain <- dfDataFinished %>% filter(train == 1) # Sample 80% of the finished jobs
dfDataFinishedTest <- dfDataFinished %>% anti_join(dfDataFinishedTrain) # Remove the train jobs from the test set

### Create function to scale the data ###
scale_data <- function(train, test, feature_range = c(0, 1)) {

  # Initialize lists to hold the scaled data and scalers
  scaled_train <- list()
  scaled_test <- list()
  scalers <- list()

  for(col in colnames(train)) {
    # For numeric columns
    if(is.numeric(train[[col]])) {
      x <- train[[col]]
      fr_min <- feature_range[1]
      fr_max <- feature_range[2]
      std_train <- ((x - min(x) ) / (max(x) - min(x)  ))
      std_test <- ((test[[col]] - min(x) ) / (max(x) - min(x)  ))
      scaled_train[[col]] <- std_train *(fr_max - fr_min) + fr_min
      scaled_test[[col]] <- std_test *(fr_max - fr_min) + fr_min
      scalers[[col]] <- list(min = min(x), max = max(x))
    }
    # For Date columns
    else if(class(train[[col]]) == "Date") {
      x <- as.integer(train[[col]])  # Convert to integer (number of days)
      scaled_train[[col]] <- x
      scaled_test[[col]] <- as.integer(test[[col]])
      scalers[[col]] <- list(type = "Date")
    }
    # For Factor columns
    else if(is.factor(train[[col]])) {
      scaled_train[[col]] <- as.integer(train[[col]])
      scaled_test[[col]] <- as.integer(test[[col]])
      scalers[[col]] <- list(type = "Factor", levels = levels(train[[col]]))
    }
    # For other types of columns
    else {
      scaled_train[[col]] <- train[[col]]
      scaled_test[[col]] <- test[[col]]
      scalers[[col]] <- list(type = "Other")
    }
  }

  return(list(scaled_train = as.data.frame(scaled_train),
              scaled_test = as.data.frame(scaled_test),
              scalers = scalers))
}


invert_scaling <- function(scaled, scaler, feature_range = c(0, 1)) {
  # Initialize list to hold the inverted data
  inverted_data <- list()
  for(col in names(scaler)) {
    # For numeric columns
    if(is.list(scaler[[col]]) && !is.null(scaler[[col]]$min)) {
      min <- scaler[[col]]$min
      max <- scaler[[col]]$max
      fr_min <- feature_range[1]
      fr_max <- feature_range[2]
      inverted <- (scaled[[col]] - fr_min) / (fr_max - fr_min) * (max - min) + min
      inverted_data[[col]] <- inverted
    }
    # For Date columns
    else if(is.list(scaler[[col]]) && scaler[[col]]$type == "Date") {
      inverted <- as.Date(scaled[[col]], origin="1970-01-01")
      inverted_data[[col]] <- inverted
    }
    # For Factor columns
    else if(is.list(scaler[[col]]) && scaler[[col]]$type == "Factor") {
      inverted <- factor(scaled[[col]], levels = seq_along(scaler[[col]]$levels), labels = scaler[[col]]$levels)
      inverted_data[[col]] <- inverted
    }
    # For other types of columns
    else {
      inverted_data[[col]] <- scaled[[col]]
    }
  }
  return(as.data.frame(inverted_data))
}


# Only keep job_no, date and numeric variables.

names(dfDataFinishedTrain)

## Run the scale_data function ##
dfDataFinishedScaled <- scale_data(dfDataFinishedTrain,dfDataFinishedTest, feature_range = c(0,1))
dfDataFinishedScaled
# dfDataFinishedTrain <- dfDataFinished$scaled_train
# dfDataFinishedTest <- dfDataFinished$scaled_test

# Scale the data
scaled_results <- scale_data(train_data, test_data)

# Invert the scaling
inverted_results <- invert_scaling(dfDataFinishedScaled$scaled_train, dfDataFinishedScaled$scalers)



## Split into dependent and independent variables ##
# Dependent variable
sDepVar <- 'final_estimate_costs'
# Independent variables
colIndepVar <- names(dfDataFinished)[!grepl(sDepVar,names(dfDataFinished))]

# Create a dataframe with only the independent variables
dfDataFinishedTrainIndep <- dfDataFinishedTrain[,colIndepVar]
dfDataFinishedTestIndep <- dfDataFinishedTest[,colIndepVar]

# Create a dataframe with only the dependent variable, date and job_no
dfDataFinishedTrainDep <- dfDataFinishedTrain[,c('date','job_no',sDepVar)]
dfDataFinishedTestDep <- dfDataFinishedTest[,c('date','job_no',sDepVar)]