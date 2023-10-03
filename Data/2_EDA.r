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

# Source GetData
source('1_FeatureEngineering.r')

rm(list=ls()[!grepl("dfData",ls())])

# Source theme_elcon
invisible(source('theme_elcon.R'))

# Date as date format ymd
dfData$date <- lubridate::ymd(dfData$date)

# Summary of Data
eda_1 <- kable(ExpData(data=dfData,type=1), format = "latex", booktabs = T, longtable = T, caption = "Summary of Dataset",
      linesep = "") %>%
      kable_styling(font_size = 9, latex_options = c("repeat_header"),repeat_header_text = "",
                    full_width = F)
writeLines(eda_1, "./Results/Tables/eda_1.tex")
eda_1

# Summary of Variables
eda_2 <- kable(ExpData(data=dfData,type=2,fun = c('mean','sd'))[,-1],
               format = "latex", booktabs = T, longtable = T, caption = "Summary of Variables",
      linesep = "") %>% kableExtra::landscape() %>%
      kable_styling(font_size = 9, bootstrap_options = c("striped", "hover", "condensed"),
                    latex_options = c("repeat_header"),repeat_header_text = "",
                    full_width = F)
writeLines(eda_2, "./Results/Tables/eda_2.tex")
eda_2
# Summary of Categorical Variables by Deparment
eda_3 <- kable(ExpCTable(dfData,Target="department",margin=1,clim=10,nlim=3,round=2,bin=NULL,per=T),
               format = "latex", booktabs = T, longtable = T, caption = "Summary of Categorical Variables by Deparment",
      linesep = "") %>%
      kable_styling(font_size = 9, latex_options = c("repeat_header"),repeat_header_text = "",
                    full_width = F)
writeLines(eda_3, "./Results/Tables/eda_3.tex")
eda_3

### Explore numeric variables ###
# Plot correlation matrix
# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
colNum <- names(dfData)[sapply(dfData, is.numeric)]

# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
dfDataX <- dfData %>%
  group_by(job_no) %>%
  mutate_at(colNum, cumsum) %>%
  filter(date == max(date))

# Remove outliers with Mahalanobis distance
# Multivariate outlier detection ------------------------------------------
multivariate_outlier <- function(df_id_plus_var,cut_off){
  # Converting to standard scores
  scaled_df <- cbind(df_id_plus_var[,1],scale(df_id_plus_var[,-1]))
  # name of ID variable into scaled variables
  colnames(scaled_df)[1] <- colnames(df_id_plus_var)[1]
  # Degrees of freedom
  deg_fre <- ncol(scaled_df)-1
  # Calculate mahalanobis distance on standard scors
  maha <- mahalanobis(scaled_df[,-1], center = F, cov = var(scaled_df[,-1]))/deg_fre
  m_outliers <- data.frame(ID = scaled_df[,1][maha>=cut_off], d2_nvar = maha[maha>=cut_off])
  # Sort outliers according to distance measure
  m_outliers <- m_outliers[order(m_outliers$d2_nvar, decreasing = T),]
  return(m_outliers)
}

mahaVar <- c('revenue',
             'costs',
             'budget_costs'
      )

# Row number of dfDataX
dfDataX$row <- seq_len(nrow(dfDataX))

lOutlier <- multivariate_outlier(df_id_plus_var = dfDataX[,c('row',mahaVar)], cut_off = 5)

# Get job numbers of outliers
lOutlier$job_no <- dfDataX[lOutlier$ID,]$job_no
lOutlier

# Dummy in dfData if outlier by job_no
dfData$outlier <- 0
dfData$outlier[dfData$job_no %in% lOutlier$job_no] <- 1

dfDataX <- dfData %>%
  group_by(job_no) %>%
  mutate_at(colNum, cumsum) %>%
  filter(date == max(date))

# Count outliers by department
tOutlier <- dfDataX %>%
                group_by(department,status,job_posting_group) %>%
                summarise(n = sum(outlier)) %>%
                arrange(desc(n))

# Plot outliers by department
ggplot(tOutlier, aes(x = reorder(department, n), y = n, fill = status)) +
  geom_bar(stat = 'identity') +
  labs(title = '', x = 'Department', y = 'Count',caption = paste0("Source: ELCON A/S")) +
  theme_elcon() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_manual(values = c(vColor[1], vColor[3])) +
  facet_wrap(~job_posting_group)
# annotate figure with method
ggsave('./Results/Figures/2_1_outlier.pdf', width = 10, height = 5)
ggsave('./Results/Presentation/2_1_outlier.svg', width = 10, height = 5)

# Summary of Data
eda_5 <- kable(ExpData(data=dfDataX,type=1), format = "latex", booktabs = T, longtable = T,
               caption = "Summary of Cross-sectional Dataset", linesep = "") %>%
      kable_styling(font_size = 9, latex_options = c("repeat_header"),repeat_header_text = "",
                    full_width = F)
writeLines(eda_5, "./Results/Tables/eda_5.tex")
eda_5


# Summary of Variables
eda_6 <- kable(ExpData(data=dfDataX,type=2,fun = c('mean','sd'))[,-1],
               format = "latex", booktabs = T, longtable = T, caption = "Summary of Cross-sectional Variables",
      linesep = "") %>% kableExtra::landscape() %>%
      kable_styling(font_size = 9, bootstrap_options = c("striped", "hover", "condensed"),
                    latex_options = c("repeat_header"),repeat_header_text = "",
                    full_width = F)
writeLines(eda_6, "./Results/Tables/eda_6.tex")
eda_6

# Summary of Categorical Variables by Deparment
eda_7 <- kable(ExpCTable(dfDataX,Target="department",margin=1,clim=10,nlim=3,round=2,bin=NULL,per=T),
               format = "latex", booktabs = T, longtable = T,
               caption = "Summary of Cross-sectional Categorical Variables by Deparment", linesep = "") %>%
      kable_styling(font_size = 9, latex_options = c("repeat_header"),repeat_header_text = "",
                    full_width = F)
writeLines(eda_7, "./Results/Tables/eda_7.tex")
eda_7

dfDataX <- dfDataX %>%
  mutate(contribution_margin = contribution / revenue,
         budget_contribution_margin = budget_contribution / budget_revenue)

# If infinite set to 0
dfDataX$contribution_margin[is.infinite(dfDataX$contribution_margin)] <- NA
dfDataX$budget_contribution_margin[is.infinite(dfDataX$budget_contribution_margin)] <- NA

# Plot histogram of contribution margin
ggplot(dfDataX, aes(x = contribution_margin)) +
  geom_histogram(bins = 50, fill = vColor[1]) +
  labs(title = '', x = 'Contribution Margin', y = 'Count',caption = paste0('Source: ELCON A/S')) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme_elcon()
ggsave('./Results/Figures/2_2_margin.pdf', width = 10, height = 5)
ggsave('./Results/Presentation/2_2_margin.svg', width = 10, height = 5)

ggplot(dfDataX, aes(x = budget_contribution_margin)) +
  geom_histogram(bins = 50, fill = vColor[1]) +
  labs(title = '', x = 'Budget Contribution Margin', y = 'Count',caption = paste0('Source: ELCON A/S')) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme_elcon()
ggsave('./Results/Figures/2_3_budget_margin.pdf', width = 10, height = 5)
ggsave('./Results/Presentation/2_3_budget_margin.svg', width = 10, height = 5)

dfCorr <- dfDataX %>%
  mutate(across(everything(), ~replace(., is.infinite(.), NA))) %>%
  select_if(is.numeric)

# Omit job_no from correlation matrix
dfCorr <- dfCorr[,!grepl('job_no',names(dfCorr))]

# Calculate correlation matrix
dfCorr <- cor(dfCorr, use = 'pairwise.complete.obs')

# Reframe data
dfCorr <- dfCorr %>%
  as.data.frame() %>%
  rownames_to_column(var = 'Var1') %>%
  gather(key = 'Var2', value = 'value', -Var1)

# Plot correlation matrix
ggplot(dfCorr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = vColor[3], mid = "white", high = vColor[4], midpoint = 0) +
  theme_elcon() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  labs(title = '', x = '', y = '',caption = paste0("Source: ELCON A/S")) +
  #geom_text(aes(label = round(value, 2)), color = 'black', size = 3) +
  theme(plot.title = element_text(hjust = 0.5))
ggsave('./Results/Figures/2_4_corr.pdf', width = 20, height = 20)
ggsave('./Results/Presentation/2_4_corr.svg', width = 20, height = 20)


# Select random job number
set.seed(156342)
sJobNo <- 'S333720'

# Filter data with selected job number
dfSample <- dfData %>% filter(job_no == sJobNo)

# Plot cost_scurve for selected job number
plotCost <- ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = costs_scurve, color = vColor[1])) +
  geom_line(aes(y = costs_cumsum, color = vColor[3])) +
  geom_line(aes(y = costs_scurve_diff, color = vColor[2])) +
    scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                         labels = c('S-curve', 'Realized', 'Difference')) +
  labs(title = '', x = 'Date', y = 'Costs',caption = paste0('Job Number: ', sJobNo,"\n Source: ELCON A/S")) +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/2_5_costs.pdf", plotCost, width = 10, height = 5)
ggsave("./Results/Presentation/2_5_costs.svg", plotCost, width = 10, height = 5)
plotCost

# Plot revenue_scurve for selected job number
plotRevenue <- ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = revenue_scurve, color = vColor[1])) +
  geom_line(aes(y = revenue_cumsum, color = vColor[3])) +
  geom_line(aes(y = revenue_scurve_diff, color = vColor[2])) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                     labels = c('S-curve', 'Realized', 'Difference')) +
  labs(title = '', x = 'Date', y = 'Revenue',
       caption = paste0('Job Number: ', sJobNo,"\n Source: ELCON A/S")) +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/2_6_revenue.pdf", plotRevenue, width = 10, height = 5)
ggsave("./Results/Presentation/2_6_revenue.svg", plotRevenue, width = 10, height = 5)
plotRevenue

# Plot contribution_scurve for selected job number
plotContrib <- ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = contribution_scurve, color = vColor[1])) +
  geom_line(aes(y = contribution_cumsum, color = vColor[3])) +
  geom_line(aes(y = contribution_scurve_diff, color = vColor[2])) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                     labels = c('S-curve', 'Realized', 'Difference')) +
  labs(title = '', x = 'Date', y = 'Contribution',
       caption = paste0('Job Number: ', sJobNo,"\n Source: ELCON A/S")) +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/2_7_contribution.pdf", plotContrib, width = 10, height = 5)
ggsave("./Results/Presentation/2_7_contribution.svg", plotContrib, width = 10, height = 5)
plotContrib

# Plot revenue_scurve_diff, costs_scurve_diff, and contribution_scurve_diff for selected job number
plotDiff <- ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = revenue_scurve_diff, color = vColor[1])) +
  geom_line(aes(y = costs_scurve_diff, color = vColor[3])) +
  geom_line(aes(y = contribution_scurve_diff, color = vColor[2])) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                     labels = c('Revenue', 'Costs', 'Contribution')) +
  labs(title = '', x = 'Date', y = 'Difference',
       caption = paste0('Job Number: ', sJobNo,"\n Source: ELCON A/S")) +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/2_8_diff.pdf", plotDiff, width = 10, height = 5)
ggsave("./Results/Presentation/2_8_diff.svg", plotDiff, width = 10, height = 5)
plotDiff

# Plot risk
plotRisk <- ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = risk, color = vColor[1])) +
  scale_color_manual(name = '', values = vColor[1],
                     labels = 'Risk') +
  labs(title = '', x = 'Date', y = 'Risk',
       caption = paste0('Job Number: ', sJobNo,"\n Source: ELCON A/S")) +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/2_10_risk.pdf", plotRisk, width = 10, height = 5)
ggsave("./Results/Presentation/2_10_risk.svg", plotRisk, width = 10, height = 5)
plotRisk

# Subplot with all plots
plotAll <- plotCost + plotRevenue + plotContrib + plotDiff + plotRisk
plotAll

# Select top 5 job with highest risk
dfRisk <- dfDataX %>%
  group_by(job_no) %>%
  summarise(risk = sum(-risk, na.rm = T)) %>%
  arrange(desc(risk)) %>%
  tail(5)

dfRisk


# Sum risk by date and department and plot time series
# Date as date format ymd
dfData$date <- lubridate::ymd(dfData$date)

dfRiskTS <- dfData %>%
  group_by(date,department) %>%
  summarise(risk = sum(risk, na.rm = T),
            contribution_scurve_diff = sum(contribution_scurve_diff, na.rm = T),
            revenue_scurve_diff = sum(revenue_scurve_diff, na.rm = T),
            costs_scurve_diff = sum(costs_scurve_diff, na.rm = T)) %>%
  arrange(date)

# Plot risk by department
ggplot(dfRiskTS, aes(x = date, y = contribution_scurve_diff, color = department)) +
  geom_line() +
  labs(title = '', x = 'Date', y = 'Risk',caption = paste0("Source: ELCON A/S")) +
    scale_color_manual(name = '', values = c(vColor[1], vColor[3])) +
  scale_x_date(date_breaks = '6 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()

plotELCON_505 <- ggplot(dfRiskTS[which(dfRiskTS$department == '505'),], aes(x = date)) +
  geom_line(aes(y = revenue_scurve_diff, color = 'revenue')) +
  geom_line(aes(y = costs_scurve_diff, color = 'costs')) +
  geom_line(aes(y = contribution_scurve_diff, color = 'contribution')) +
  geom_line(aes(y = risk, color = 'risk')) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2], vColor[5]),
                     labels = c('Revenue', 'Costs', 'Contribution','Risk')) +
  labs(title = '505', x = 'Date', y = 'Difference',
       caption = paste0("Source: ELCON A/S")) +
  scale_x_date(date_breaks = '12 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
plotELCON_515 <- ggplot(dfRiskTS[which(dfRiskTS$department == '515'),], aes(x = date)) +
  geom_line(aes(y = revenue_scurve_diff, color = 'revenue')) +
  geom_line(aes(y = costs_scurve_diff, color = 'costs')) +
  geom_line(aes(y = contribution_scurve_diff, color = 'contribution')) +
  geom_line(aes(y = risk, color = 'risk')) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2], vColor[5]),
                     labels = c('Revenue', 'Costs', 'Contribution','Risk')) +
  labs(title = '515', x = 'Date', y = 'Difference',
       caption = paste0("Source: ELCON A/S")) +
  scale_x_date(date_breaks = '12 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()

plotELCON_505 + plotELCON_515




