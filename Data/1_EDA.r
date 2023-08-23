library(SmartEDA) 
library(arrow)
library(ggplot2)
library(ggthemes)
library(ggbreak)
library(svglite)
library(dplyr)
library(tidyr)
library(texreg)
library(xtable)

rm(list=ls())

# Source GetData
#source('0_GetData.r')
# Source theme_elcon
invisible(source('theme_elcon.R'))

dir <- "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
setwd(dir)

# Get directory of file
dfData <- read_parquet("dfData.parquet")

# Order by date
dfData <- dfData %>% arrange(date)

# use ExpReport
# ExpReport(dfData, op_file = 'ExploratoryDataAnalysis.html')


eda_1 <- xtable(ExpData(data=dfData,type=1),
            caption= "Summary of Dataset", 
            align=c("l","l","c"), include.rownames=F)
print(eda_1,file="./Results/Tables/eda_1.tex",append=F,
      caption.placement="top")
eda_1

eda_2 <- xtable(ExpData(data=dfData,type=2)[,-1],
              caption= "Summary of Variables", include.rownames=F)
print(eda_2,file="./Results/Tables/eda_2.tex",append=F,
      caption.placement="top",floating = T, floating.environment = "sidewaystable")
eda_2

eda_3 <- xtable(ExpCTable(dfData,Target="department",margin=1,clim=10,nlim=3,round=2,bin=NULL,per=T),
                caption= "Summary of Categorical Variables by Deparment", include.rownames=F)
print(eda_3,file="./Results/Tables/eda_3.tex",append=F,
      caption.placement="top",floating = T, floating.environment = "sidewaystable")
eda_3

### Explore numeric variables ###
# Plot correlation matrix
# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
# colNum <- c('revenue','costs','costs_of_labor','costs_of_materials','other_costs','contribution','estimated_revenue','estimated_contribution')
colNum <- names(dfData)[sapply(dfData, is.numeric)]
# Calculate days until end of job
dfData <- dfData %>%
  mutate(days_until_end = end_date - date)
dfData$days_until_end[dfData$days_until_end < 0] <- 0
dfData$days_until_end <- as.numeric(dfData$days_until_end)

# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
dfDataX <- dfData %>%
  group_by(job_no) %>%
  mutate_at(colNum, cumsum) %>%
  filter(date == max(date))

# Calculate contribution margin
dfDataX$contribution_margin <- dfDataX$contribution / dfDataX$revenue

# Plot histogram of contribution margin
ggplot(dfDataX, aes(x = contribution_margin)) +
  geom_histogram(bins = 50, fill = vColor[1]) +
  labs(title = '', x = 'Contribution Margin', y = 'Count') +
  theme_elcon()

# Plot histogram of colCum variables in facet
dfDataX <- dfDataX %>%
  gather(key = 'variable', value = 'value',  c(colNum,'days_until_end'))

ggplot(dfDataX, aes(x = value)) +
    geom_histogram(bins = 50, fill = vColor[1]) +
    labs(title = '', x = 'Value', y = 'Count') +
    scale_x_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
    facet_wrap(~variable, scales = 'free_x') +
    theme_elcon()
ggsave("./Results/Figures/eda_3.svg", dpi = 320)

# Sample
# Select random job number
set.seed(1542)
sJobNo <- sample(dfData$job_no,1)
# Filter data with selected job number
dfSample <- dfData %>% filter(job_no == sJobNo)


# Plot cumulative contribution
ggplot(dfSample, aes(x = date, y = contribution)) +
  geom_line() +
  labs(title = paste0("Cumulative contribution for ", dfSample$job_no,' - ',dfSample$description),
       subtitle = "Contribution = Revenue - Costs",
       x = "Date",
       y = "Cumulative contribution") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red")