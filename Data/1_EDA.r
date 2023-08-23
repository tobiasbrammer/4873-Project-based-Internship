library(SmartEDA) 
library(arrow)
library(ggplot2)
library(ggthemes)
library(svglite)
library(dplyr)
library(tidyr)
library(texreg)
library(xtable)

rm(list=ls())

# Source GetData
source('0_GetData.r')
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


### Explore numeric variables ###

# Plot correlation matrix
# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
# colNum <- c('revenue','costs','costs_of_labor','costs_of_materials','other_costs','contribution','estimated_revenue','estimated_contribution')
colNum <- names(dfData)[sapply(dfData, is.numeric)]
# Calculate days until end of job
dfData <- dfData %>%
  group_by(job_no) %>%
  mutate(days_until_end = end_date - date)
dfData$days_until_end[dfData$days_until_end < 0] <- 0
dfData$days_until_end <- as.numeric(dfData$days_until_end)

# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
dfDataX <- dfData %>%
  group_by(job_no) %>%
  mutate_at(colNum, cumsum) %>%
  filter(date == max(date))

# Plot histogram of colCum variables in facet
dfDataX <- dfDataX %>%
  gather(key = 'variable', value = 'value',  c(colNum,'days_until_end'))

ggplot(dfDataX, aes(x = value)) +
    geom_histogram(bins = 50, fill = vColor[1]) +
    labs(title = '', x = 'Value', y = 'Count') +
    scale_x_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
    facet_wrap(~variable, scales = 'free') +
    theme_elcon()
ggsave("./Results/Figures/eda_3.svg", dpi = 320)

# Reframe dfDataX
dfDataX <- dfData %>%
  group_by(job_no) %>%
  mutate_at(colNum, cumsum) %>%
  filter(date == max(date))

# Get all character variables
colChar <- names(dfData)[sapply(dfData, is.character)]
colCor <- names(dfData)[sapply(dfData, is.numeric)]

# Calculate correlation matrix of all except character variables
CorX <- dfDataX %>%
  select(colCor) %>%
  cor()