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

# Convert to cross-sectional
colNum <- names(dfData)[sapply(dfData, is.numeric)]
colNum <- colNum[!grepl("_rate",colNum)]
colNum <- names(dfData)[sapply(dfData, is.numeric)]
colNum <- colNum[!grepl("_share",colNum)]
colNum <- colNum[!grepl("_rate",colNum)]
colNum <- colNum[!grepl("_margin",colNum)]
colCumSum <- colNum[!grepl("budget_",colNum)]


# Define the complete model including both measurement and structural models
model <- "
# Reflective Measurement model
contribution_margin =~ revenue + costs
revenue =~ revenue_budget_share + scurve
costs =~ costs_of_labor_share + costs_of_materials_share
costs_of_materials =~ overrun_risk + budget_costs
costs_of_labor =~ billable_rate_dep + billable_hours_qty
billable_rate_dep =~ illness_rate_dep + internal_rate_dep
risk =~ efficiency_risk + overrun_risk
illness_rate_dep <~ illness_rate_dep
efficiency_risk <~ efficiency_risk
overrun_risk <~ overrun_risk

# Structural model
contribution_margin ~ revenue + costs + risk
billable_rate_dep ~ illness_rate_dep + efficiency_risk
risk ~ efficiency_risk + overrun_risk
"

# Estimate the model
results <- csem(.data = dfDataX, .model = model)

# Summary of results
summary(results)

