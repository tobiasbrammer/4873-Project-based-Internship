# update.packages(checkBuilt=TRUE, ask=FALSE)
library(dplyr)
library(c)
library(tidyr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(arrow)
library(foreach)
library(doParallel)
library(doSNOW)
# Get directory of file
dfData <- read_parquet("dfData.parquet")
# Set date as date
dfData$date <- as.Date(dfData$date, format = "%YYYY-%dd-%MM")
dfData$end_date <- as.Date(dfData$end_date, format = "%YYYY-%dd-%MM")

# Order by date
dfData <- dfData %>% arrange(date)

# Define columns to calculate cumulative values
colCum <- c('revenue','costs','costs_of_labor','costs_of_materials','other_costs','contribution','estimated_revenue','estimated_contribution')

# Define color palette
col <- c('#006e64','#ffbb00','#c17150','#1e8c82','#734848','#dcfae9')

# Group by job number and calculate cumulative values
dfData_cumsum <- dfData %>% group_by(job_no) %>% mutate_at(colCum, cumsum)

# Select random job number
set.seed(156342)
sJobNo <- sample(dfData_cumsum$job_no,1)
# Filter data with selected job number
dfSample <- dfData_cumsum %>% filter(job_no == sJobNo)


# Plot cumulative contribution
ggplot(dfSample, aes(x = date, y = )) +
  geom_line() +
  labs(title = paste0("Cumulative contribution for ", dfSample$job_no,' - ',dfSample$description),
       subtitle = "Contribution = Revenue - Costs",
       x = "Date",
       y = "Cumulative contribution") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red")

# Plot cumulative estimated contribution
ggplot(dfSample, aes(x = date, y = estimated_contribution)) +
  geom_line() +
    labs(title = paste0("Cumulative estimated contribution for ", dfSample$job_no,' - ',dfSample$description),
         subtitle = "Estimated contribution = Estimated revenue - Costs",
         x = "Date",
         y = "Cumulative estimated contribution") +
    # Format y-axis with thousands separator and decimal point
    scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
    theme_economist() +
    theme(plot.title = element_text(hjust = 0.5)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red")

# Plot cumulative contribution and cumulative estimated contribution
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = contribution, color = "Cumulative contribution")) +
  geom_line(aes(y = estimated_contribution, color = "Cumulative estimated contribution")) +
  labs(title = paste0("Cumulative contribution for ", dfSample$job_no,' - ',dfSample$description),
       subtitle = "Contribution = Revenue - Costs",
       x = "Date",
       y = "Cumulative contribution") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) +
  guides(color=guide_legend(title=NULL))

# Plot cumulative revenue
ggplot(dfSample, aes(x = date, y = revenue)) +
  geom_line() +
  labs(title = paste0("Cumulative revenue for ", dfSample$job_no,' - ',dfSample$description),
       x = "Date",
       y = "Cumulative revenue") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red")

# Plot cumulative costs_of_labor and cumulative costs_of_materials
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = costs_of_labor, color = "Cumulative costs of labor")) +
  geom_line(aes(y = costs_of_materials, color = "Cumulative costs of materials")) +
  labs(title = paste0("Cumulative costs of labor and materials for ", dfSample$job_no,' - ',dfSample$description),
       x = "Date",
       y = "Cumulative costs") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) +
  guides(color=guide_legend(title=NULL))

############################# Theoretical s-curve #############################
dBudget <- tail(dfSample$budget_costs,1)
dBudgetLabor <- tail(dfSample$budget_costs,1) * 
                sum(dfSample$costs_of_labor)/sum(dfSample$costs)
dBudgetMat <- tail(dfSample$budget_costs,1) * 
              sum(dfSample$costs_of_materials)/sum(dfSample$costs)

# Calculate the days between start and end of project
dStart <- as.numeric(head(dfSample$date,1))
dEnd <- as.numeric(tail(dfSample$end_date,1))
# Calculate the number of days between start and end of project
nDays <- as.numeric(dEnd - dStart)

# Iteratively calculate the theoretical s-curve for different values of dSkew and dPeak until the 
# theoretical s-curve fits the actual s-curve

# Define the function to calculate the sum of squared errors
SSE <- function(dSkew, dPeak) {
  # Calculate the theoretical s-curves according to \frac{1-\cos(\pi t^{s})+pt^{s}}{2+p}
  dfSample <- dfSample %>%
    mutate(date_numeric = as.numeric(date), # convert date to numeric
           TheoreticalScurveLabor = dBudgetLabor * (1-cos(pi*((date_numeric-dStart)/nDays)^(dSkew)) + dPeak *((date_numeric-dStart)/nDays)^(dSkew))/(2 + dPeak),
           TheoreticalScurveMat = dBudgetMat * (1-cos(pi*((date_numeric-dStart)/nDays)^(dSkew)) + dPeak *((date_numeric-dStart)/nDays)^(dSkew))/(2 + dPeak),
           TheoreticalScurve = dBudget * (1-cos(pi*((date_numeric-dStart)/nDays)^(dSkew)) + dPeak *((date_numeric-dStart)/nDays)^(dSkew))/(2 + dPeak)) # calculate theoretical s-curve
  
  # Calculate the sum of squared errors
  SSE <- sum((dfSample$TheoreticalScurve - dfSample$costs)^2)
  return(SSE)
}

# For each value of dSkew, calculate the sum of squared errors for different values of dPeak
dSkew <- seq(-5, 5, 0.1) # define the range of dSkew
dPeak <- seq(-5, 5, 0.1) # define the range of dPeak

# Register the parallel backend with the number of cores you want to use
cl <- makeCluster(detectCores() - 1) # using all cores minus one
registerDoSNOW(cl)

# progress bar ------------------------------------------------------------
library(progress)

pb <- progress_bar$new(
  format = "current = :var [:bar] :elapsed | eta: :eta",
  total = length(dSkew) ,    # 100 
  width = 60)

progress_var <- dSkew  # token reported in progress bar

# allowing progress bar to be used in foreach -----------------------------
progress <- function(n){
  pb$tick(tokens = list(var = progress_var[n]))
} 

opts <- list(progress = progress)


# Initialize the matrix to store the sum of squared errors
SSEMatrix <- matrix(NA, nrow = length(dSkew), ncol = length(dPeak))

# Calculate the sum of squared errors for each combination of dSkew and dPeak
results <- foreach(i = 1:length(dSkew), .combine = 'cbind', .packages = 'tidyverse',
                   .export = "dfSample", 
                   .options.snow = opts) %dopar% {
  temp <- numeric(length(dPeak))
  for (j in 1:length(dPeak)) {
    if (i <= nrow(SSEMatrix) && j <= ncol(SSEMatrix)) { # check if i and j are within bounds
      temp[j] <- SSE(dSkew[i], dPeak[j])
    }
  }
  temp
}

# Transpose the results to get the SSEMatrix
SSEMatrix <- t(results)

stopCluster(cl)

# Find the minimum sum of squared errors and the corresponding values of dSkew and dPeak
minSSE <- min(SSEMatrix, na.rm = TRUE)
minSSEIndex <- which(SSEMatrix == minSSE, arr.ind = TRUE)
dSkewMinSSE <- dSkew[minSSEIndex[1]]
dPeakMinSSE <- dPeak[minSSEIndex[2]]

# dSkewMinSSE <- 0.79
# dPeakMinSSE <- 1.1

# Calculate the theoretical s-curves according to \frac{1-\cos(\pi t^{s})+pt^{s}}{2+p} with the values of dSkew and dPeak that minimize the sum of squared errors
dfSample <- dfSample %>%
  mutate(date_numeric = as.numeric(date), # convert date to numeric
         TheoreticalScurveLabor = dBudgetLabor * (1-cos(pi*((date_numeric-dStart)/nDays)^(dSkewMinSSE)) + dPeakMinSSE *((date_numeric-dStart)/nDays)^(dSkewMinSSE))/(2 + dPeakMinSSE),
         TheoreticalScurveMat = dBudgetMat * (1-cos(pi*((date_numeric-dStart)/nDays)^(dSkewMinSSE)) + dPeakMinSSE *((date_numeric-dStart)/nDays)^(dSkewMinSSE))/(2 + dPeakMinSSE),
         TheoreticalScurve = dBudget * (1-cos(pi*((date_numeric-dStart)/nDays)^(dSkewMinSSE)) + dPeakMinSSE *((date_numeric-dStart)/nDays)^(dSkewMinSSE))/(2 + dPeakMinSSE))

# Plot the theoretical s-curves
# Varer
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = TheoreticalScurveMat, color = "Theoretical S-curve")) +
  geom_line(aes(y = costs_of_materials, color = "Costs of Materials")) +
  geom_ribbon(aes(ymin = costs_of_materials, ymax = TheoreticalScurveMat), fill = "grey", alpha = 0.5) +
  labs(title = paste0("Theoretical S-curve for Costs of Materials"),
       subtitle = paste0(dfSample$job_no,' - ',dfSample$description),
       x = "date",
       y = "Cumulative Costs of Materials") +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))

# Ressources 
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = TheoreticalScurveLabor, color = "Theoretical S-curve")) +
  geom_line(aes(y = costs_of_labor, color = "Costs of Labor")) +
  geom_ribbon(aes(ymin = costs_of_labor, ymax = TheoreticalScurveLabor), fill = "grey", alpha = 0.5) +
  labs(title = paste0("Theoretical S-curve for Costs of Labor"),
       subtitle = paste0(dfSample$job_no,' - ',dfSample$description),
       x = "date",
       y = "Costs of Labor") +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))

# Total costs
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = TheoreticalScurve, color = "Theoretical S-curve")) +
  geom_line(aes(y = costs, color = "Costs")) +
  geom_ribbon(aes(ymin = costs, ymax = TheoreticalScurve), fill = "grey", alpha = 0.5) +
  labs(title = paste0("Theoretical S-curve for Costs"),
       subtitle = paste0(dfSample$job_no,' - ',dfSample$description),
       x = "date",
       y = "Cumulative Costs") +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))


