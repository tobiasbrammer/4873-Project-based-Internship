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
library(cSEM)

# Source GetData
source('2_EDA.r')
rm(list=ls()[!grepl("dfData",ls())])
invisible(source('theme_elcon.R'))
rm(list=ls()[!grepl("vColor|theme_elcon|dfData",ls())])
beep()

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


### Predict total_costs with a naïve linear model ###
# Preprocess data
# Omit train and wip
dfDataFinishedTrain <- dfDataFinishedTrain[,!grepl("train|wip|outlier",names(dfDataFinishedTrain))]
dfDataFinishedTest <- dfDataFinishedTest[,!grepl("train|wip|outlier",names(dfDataFinishedTest))]

# Create a linear model
lmTotalCosts <- lm(total_costs ~ ., data = dfDataFinishedTrain)

# Get the number of unique values for each variable
dfDataDesc <- dfDataFinishedTrain %>%
  summarise_all(n_distinct) %>%
  gather(variable, value) %>%
  arrange((value))

# Omit variables with only 1 unique value.
dfDataDesc <- dfDataDesc %>%
  filter(value <= 1)

# Omit variables in dfDataDesc$variable from the linear model
lmTotalCosts <- lm(total_costs ~ ., data = dfDataFinishedTrain[,!grepl(paste0(dfDataDesc$variable,collapse = "|"),names(dfDataFinishedTrain))])

# Predict total_costs
lmTotalCosts$xlevels <- union(lmTotalCosts$xlevels, levels(dfDataFinishedTest))
dfDataFinishedTest$pred_costs <- stats::predict(lmTotalCosts, newdata = dfDataFinishedTest)

# Calculate the mean absolute percentage error. If denominator is 0, set to NA
dfDataFinishedTest$mape <- abs(dfDataFinishedTest$total_costs - dfDataFinishedTest$pred_costs)/dfDataFinishedTest$total_costs
dfDataFinishedTest$pe <- abs(dfDataFinishedTest$total_costs - dfDataFinishedTest$pred_costs)
dfDataFinishedTest$mape[is.nan(dfDataFinishedTest$mape)] <- NA

# Plot mean absolute prediction error by date
dfDataFinishedTest %>%
  group_by(date) %>%
  summarise(mape = mean(mape, na.rm = T)) %>%
  ggplot(aes(x = date, y = mape)) +
  geom_line() +
  geom_point() +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  theme_elcon() +
  labs(title = "Mean absolute prediction error by date",
       subtitle = "Linear model",
       x = "Date",
       y = "Mean absolute prediction error") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Plot job S150469
dfDataDesc

# Top 10 highest pe
dfDataFinishedTest %>%
  arrange(desc(pe)) %>%
  head(100) %>%
  dplyr::select(job_no, date, total_costs, pred_costs, mape,pe)

# Calculate the mean absolute percentage error
paste0("The mean absolute percentage error is ",
       round(mean(dfDataFinishedTest$mape, na.rm = T)*100,2),
       "%.")
beep(2)