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

rm(list=ls())

# Source GetData
source('1_FeatureEngineering.r')


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

mahaVar <- c('revenue','costs_of_labor','costs_of_materials','other_costs',
             'estimated_revenue','sales_estimate_contribution',
             'production_estimate_contribution',
             'final_estimate_contribution')

# Row number of dfDataX
dfDataX$row <- 1:nrow(dfDataX)

lOutlier <- multivariate_outlier(df_id_plus_var = dfDataX[,c('row',mahaVar)], cut_off = 6)

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
  labs(title = '', x = 'Department', y = 'Count',caption = "Source: ELCON A/S") +
  theme_elcon() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_fill_manual(values = c(vColor[1], vColor[3])) +
  facet_wrap(~job_posting_group)
# annotate figure with method
ggsave('./Results/Figures/outlier.pdf', width = 10, height = 5)

eda_5 <- xtable(ExpData(data=dfDataX,type=1),
                caption= "Summary of Cross-sectional Dataset", 
                align=c("l","l","c"), include.rownames=F)
print(eda_5,file="./Results/Tables/eda_5.tex",append=F,
      caption.placement="top")
eda_5

eda_6 <- xtable(ExpData(data=dfDataX,type=2,fun = c('mean'))[,-1],
                caption= "Summary of Cross-sectional Variables", include.rownames=F)
print(eda_6,file="./Results/Tables/eda_6.tex",append=F,
      caption.placement="top",floating = T, floating.environment = "sidewaystable")
eda_6

eda_7 <- xtable(ExpCTable(dfDataX,Target="department",margin=1,clim=10,nlim=3,round=2,bin=NULL,per=T),
                caption= "Summary of Cross-sectional Categorical Variables by Deparment", include.rownames=F)
print(eda_7,file="./Results/Tables/eda_7.tex",append=F,
      caption.placement="top",floating = T, floating.environment = "sidewaystable")
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
  labs(title = '', x = 'Contribution Margin', y = 'Count',caption = "Source: ELCON A/S") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme_elcon()
ggsave('./Results/Figures/margin.pdf', width = 10, height = 5)

ggplot(dfDataX, aes(x = budget_contribution_margin)) +
  geom_histogram(bins = 50, fill = vColor[1]) +
  labs(title = '', x = 'Budget Contribution Margin', y = 'Count',caption = "Source: ELCON A/S") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  theme_elcon()
ggsave('./Results/Figures/budget_margin.pdf', width = 10, height = 5)

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
  labs(title = '', x = '', y = '',caption = "Source: ELCON A/S") +
  #geom_text(aes(label = round(value, 2)), color = 'black', size = 3) +
  theme(plot.title = element_text(hjust = 0.5))
ggsave('./Results/Figures/corr.pdf', width = 20, height = 20)
ggsave('./Results/Presentation/corr.svg', width = 20, height = 20)

# Plot histogram of colCum variables in facet
dfDataXfacet <- dfDataX %>%
  gather(key = 'variable', value = 'value',  c(colNum,'days_until_end'))

ggplot(dfDataXfacet, aes(x = value)) +
    geom_histogram(bins = 50, fill = vColor[1]) +
    labs(title = '', x = 'Value', y = 'Count',caption = "Source: ELCON A/S") +
    scale_x_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
    facet_wrap(~variable, scales = 'free') +
    theme_elcon()
ggsave("./Results/Figures/facet.pdf", width = 20, height = 20)
ggsave("./Results/Presentation/facet.svg", width = 20, height = 20)

# Select random job number
set.seed(156342)
sJobNo <- 'S283202'
sJobNo <- sample(dfData$job_no,1)
# Filter data with selected job number
dfSample <- dfData %>% filter(job_no == sJobNo)

# Plot cost_scurve for selected job number
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = costs_scurve, color = vColor[1])) +
  geom_line(aes(y = costs_cumsum, color = vColor[3])) +
  geom_line(aes(y = costs_scurve_diff, color = vColor[2])) +
    scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                         labels = c('S-curve', 'Realized', 'Difference')) +
  labs(title = paste0('Costs for Job Number: ', sJobNo), x = 'Date', y = 'Costs',caption = "Source: ELCON A/S") +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/costs.pdf", width = 10, height = 5)
ggsave("./Results/Presentation/costs.svg", width = 10, height = 5)

# Plot revenue_scurve for selected job number
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = revenue_scurve, color = vColor[1])) +
  geom_line(aes(y = revenue_cumsum, color = vColor[3])) +
  geom_line(aes(y = revenue_scurve_diff, color = vColor[2])) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                     labels = c('S-curve', 'Realized', 'Difference')) +
  labs(title = paste0('Revenue for Job Number: ', sJobNo), x = 'Date', y = 'Revenue',
       caption = "Source: ELCON A/S") +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/revenue.pdf", width = 10, height = 5)
ggsave("./Results/Presentation/revenue.svg", width = 10, height = 5)

# Plot contribution_scurve for selected job number
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = contribution_scurve, color = vColor[1])) +
  geom_line(aes(y = contribution_cumsum, color = vColor[3])) +
  geom_line(aes(y = contribution_scurve_diff, color = vColor[2])) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                     labels = c('S-curve', 'Realized', 'Difference')) +
  labs(title = paste0('Contribution for Job Number: ', sJobNo), x = 'Date', y = 'Contribution',
       caption = "Source: ELCON A/S") +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/contribution.pdf", width = 10, height = 5)
ggsave("./Results/Presentation/contribution.svg", width = 10, height = 5)


# Plot revenue_scurve_diff, costs_scurve_diff, and contribution_scurve_diff for selected job number
ggplot(dfSample, aes(x = date)) +
  geom_line(aes(y = revenue_scurve_diff, color = vColor[1])) +
  geom_line(aes(y = costs_scurve_diff, color = vColor[3])) +
  geom_line(aes(y = contribution_scurve_diff, color = vColor[2])) +
  scale_color_manual(name = '', values = c(vColor[1], vColor[3], vColor[2]),
                     labels = c('Revenue', 'Costs', 'Contribution')) +
  labs(title = paste0('Difference between S-curve and Realized for Job Number: ', sJobNo), x = 'Date', y = 'Difference',
       caption = "Source: ELCON A/S") +
  scale_x_date(date_breaks = '3 months', date_labels = '%m %Y') +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_elcon()
ggsave("./Results/Figures/diff.pdf", width = 10, height = 5)
ggsave("./Results/Presentation/diff.svg", width = 10, height = 5)
