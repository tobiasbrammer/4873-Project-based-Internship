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
set.seed(6056)
source('4_LM.R')
rm(list=ls()[!grepl("dfData",ls())])
invisible(source('theme_elcon.R'))
rm(list=ls()[!grepl("vColor|theme_elcon|dfData|lVars",ls())])

beep()

## In this script we will use the data from the previous script to make a PLS model. ##
## We will use the caret package to make the model. ##

# NA in dfDataFinishedTrain
dfDataFinishedTrain <- dfDataFinishedTrain[complete.cases(dfDataFinishedTrain),]

plsTotalCostsAll <- train(total_costs ~ .,
                          data = dfDataFinishedTrain,
                          method = "pls",
                          scale = TRUE,
                          tuneLength = 10,
  trControl = trainControl(method = "cv", number = 10, verboseIter = F),
  preProcess = c("center", "scale"),
  metric = "RMSE")

