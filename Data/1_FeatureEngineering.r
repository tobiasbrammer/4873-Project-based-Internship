library(SmartEDA)
library(arrow)
library(tibble)
library(ggplot2)
library(ggthemes)
library(ggbreak)
library(svglite)
library(dplyr)
library(tidyr)
library(texreg)
library(xtable)
library(beepr)
library(readxl)
library(MASS)
library(purrr)
library(arrow)

rm(list=ls())

# Source GetData.
source('0_GetData.r')

# Source theme_elcon
invisible(source('theme_elcon.R'))

dir <- "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
setwd(dir)

dfData <- arrow::read_parquet("./dfData.parquet")

# Order by date
dfData <- dfData %>% arrange(date)

# Summary of Data
eda_1 <- xtable(ExpData(data=dfData,type=1),
            caption= "Summary of Dataset",
            align=c("l","l","c"), include.rownames=F)
print(eda_1,file="./Results/Tables/eda_1.tex",append=F,
      caption.placement="top")
eda_1

# Summary of Variables
eda_2 <- xtable(ExpData(data=dfData,type=2,fun = c('mean'))[,-1],
              caption= "Summary of Variables", include.rownames=F)
print(eda_2,file="./Results/Tables/eda_2.tex",append=F,
      caption.placement="top",floating = T, floating.environment = "sidewaystable")
eda_2

# Summary of Categorical Variables
eda_3 <- xtable(ExpCTable(dfData,Target="department",margin=1,clim=10,nlim=3,round=2,bin=NULL,per=T),
                caption= "Summary of Categorical Variables by Deparment", include.rownames=F)
print(eda_3,file="./Results/Tables/eda_3.tex",append=F,
      caption.placement="top",floating = T, floating.environment = "sidewaystable")
eda_3

### Explore numeric variables ###
# Plot correlation matrix
# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
colNum <- names(dfData)[sapply(dfData, is.numeric)]
colNum <- colNum[!grepl("_share",colNum)]
colNum <- colNum[!grepl("_rate",colNum)]
colCumSum <- colNum[!grepl("budget_",colNum)]


# For each variable in colCumSum create a new variable with cumulative sum by job_no. Name of new variable is the same as the old variable with "_cumsum" added
dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(across(all_of(colCumSum), cumsum, .names = "{.col}_cumsum")) %>%
  ungroup()

# Calculate days until end of job
dfData <- dfData %>%
  mutate(days_until_end = end_date - date)

dfData$days_until_end[dfData$days_until_end < 0] <- 0
dfData$days_until_end <- as.numeric(dfData$days_until_end)

##### Feature engineering #####
# Difference between sales_estimate and production_estimate
dfData$sales_production_diff <- dfData$sales_estimate_contribution - dfData$production_estimate_contribution

# Difference between sales_estimate and final_estimate
dfData$sales_final_diff <- dfData$sales_estimate_contribution - dfData$final_estimate_contribution

# Difference between realized contribution and sales_estimate_contribution
dfData$contribution_sales_diff <- dfData$contribution - dfData$sales_estimate_contribution

# Difference between realized contribution and production_estimate_contribution
dfData$contribution_production_diff <- dfData$contribution - dfData$production_estimate_contribution

# Difference between realized contribution and final_estimate_contribution
dfData$contribution_final_diff <- dfData$contribution - dfData$final_estimate_contribution

# Change in sales_estimate_contribution from previous observation. Group by job_no and order by date
dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(sales_estimate_contribution_change = sales_estimate_contribution - lag(sales_estimate_contribution),
         production_estimate_contribution_change = production_estimate_contribution - lag(production_estimate_contribution),
         final_estimate_contribution_change = final_estimate_contribution - lag(final_estimate_contribution))

dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(sales_estimate_contribution_change = ifelse(is.na(sales_estimate_contribution_change), 0, sales_estimate_contribution_change),
         production_estimate_contribution_change = ifelse(is.na(production_estimate_contribution_change), 0, production_estimate_contribution_change),
         final_estimate_contribution_change = ifelse(is.na(final_estimate_contribution_change), 0, final_estimate_contribution_change))

# Calculate revenue according to s-curve
dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(days_since_start = date - min(date),
         total_days = max(end_date) - min(date), # make total_days a difftime object
         progress = as.numeric(days_since_start) / as.numeric(total_days)) %>%
         ungroup() %>% # S-curve according to (1/(1+exp(-kx))^a.
         mutate(
         scurve = (1/(1+exp(-6*(progress-0.5))))^2,
         revenue_scurve = scurve * budget_revenue,
         costs_scurve = scurve * budget_costs,
         revenue_scurve_diff = revenue_scurve - revenue_cumsum,
         costs_scurve_diff = costs_scurve - costs_cumsum,
         contribution_scurve = scurve * contribution,
         contribution_scurve_diff = contribution_scurve - contribution_cumsum,
         sales_estimate_contribution_scurve = sales_estimate_contribution * scurve,
         production_estimate_contribution_scurve = production_estimate_contribution * scurve,
         final_estimate_contribution_scurve = final_estimate_contribution * scurve,
         sales_estimate_contribution_scurve_diff = sales_estimate_contribution_scurve - sales_estimate_contribution_cumsum,
         production_estimate_contribution_scurve_diff = production_estimate_contribution_scurve - production_estimate_contribution_cumsum,
         final_estimate_contribution_scurve_diff = final_estimate_contribution_scurve - final_estimate_contribution
         )

# Read xlxs file with cvr, equity ratio and quick ratio
dfCvr <- read_excel("./.AUX/Cvr.xlsx")
dfCvr <- dfCvr %>%
  na.omit()
# Drop column 2
dfCvr <- dfCvr[,-2]
# Rename columns
colnames(dfCvr) <- c("cvr","customer_equity_ratio","customer_quick_ratio")
# Set cvr as character
dfCvr$cvr <- as.character(dfCvr$cvr)

# Replace . with , in cvr
dfCvr$customer_equity_ratio <- as.numeric(gsub("\\,","\\.",dfCvr$customer_equity_ratio))
dfCvr$customer_quick_ratio <- as.numeric(gsub("\\,","\\.",dfCvr$customer_quick_ratio))

# Divide by 100
dfCvr$customer_equity_ratio <- dfCvr$customer_equity_ratio/100
dfCvr$customer_quick_ratio <- dfCvr$customer_quick_ratio/100

# Join dfCvr on dfData
dfData <- left_join(dfData,dfCvr,by="cvr")

# Impute missing values in customer_equity_ratio and customer_quick_ratio with mean
dfData$customer_equity_ratio[is.na(dfData$customer_equity_ratio)] <- mean(dfData$customer_equity_ratio, na.rm = T)
dfData$customer_quick_ratio[is.na(dfData$customer_quick_ratio)] <- mean(dfData$customer_quick_ratio, na.rm = T)

# Replace NaN & Inf with NA
dfData <- replace(dfData, is.infinite(as.matrix(dfData)), NA)
dfData <- replace(dfData, is.nan(as.matrix(dfData)), NA)

# Replace NaN with NA
dfData <- replace(dfData, is.nan(as.matrix(dfData)), NA)
# Replace Inf with NA
dfData <- replace(dfData, is.infinite(as.matrix(dfData)), NA)

# Define risk as the log of the ratio between the realized and the S-curve. Omit NaN
dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(
    labor_cost_risk_measure = (costs_scurve_diff * costs_of_labor / costs),
    material_cost_risk_measure = (costs_scurve_diff * costs_of_materials / costs)
  ) %>%
  do({
    if (any(is.na(.$contribution_scurve_diff)) || any(is.na(.$contribution_cumsum))) {
      data.frame(., risk = NA_real_)
    } else {
      model <- lm(contribution_scurve_diff ~ revenue_scurve_diff + costs_scurve_diff + 1
      , data = .)
      data.frame(., risk = -model$residuals)
    }
  })

dfData$risk <- (dfData$risk)^(1/2)

library(ExPanDaR)

prepare_scatter_plot(dfData, x="risk", y="contribution", color="department", loess = 1)

cor(dfData$risk,dfData$contribution,use="complete.obs")
