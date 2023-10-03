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
library(mice)
library(data.table)
library(purrr)
library(RecordLinkage)
library(stringr)
library(tm)

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

### Explore numeric variables ###
# Plot correlation matrix
# Transform to cross-sectional data using group by job_no, calculate cumulative numbers, and select latest date
colNum <- names(dfData)[sapply(dfData, is.numeric)]
colNum <- colNum[!grepl("_share",colNum)]
colNum <- colNum[!grepl("_rate",colNum)]
colNum <- colNum[!grepl("_ratio",colNum)]
colNum <- colNum[!grepl("_margin",colNum)]
#colNum <- colNum[!grepl("_estimate",colNum)]
colNum <- colNum[!grepl("_cumsum",colNum)]
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
         contribution_scurve = scurve * (budget_revenue - budget_costs),
         contribution_scurve_diff = contribution_scurve - contribution_cumsum
         )

# # Read xlxs file with cvr, equity ratio and quick ratio
# dfCvr <- read_excel("./.AUX/Cvr.xlsx")
# dfCvr <- dfCvr %>%
#   na.omit()
# # Drop column 2
# dfCvr <- dfCvr[,-2]
# # Rename columns
# colnames(dfCvr) <- c("cvr","customer_equity_ratio","customer_quick_ratio")
# # Set cvr as character
# dfCvr$cvr <- as.character(dfCvr$cvr)
#
# # Replace . with , in cvr
# dfCvr$customer_equity_ratio <- as.numeric(gsub("\\,","\\.",dfCvr$customer_equity_ratio))
# dfCvr$customer_quick_ratio <- as.numeric(gsub("\\,","\\.",dfCvr$customer_quick_ratio))
#
# # Divide by 100
# dfCvr$customer_equity_ratio <- dfCvr$customer_equity_ratio/100
# dfCvr$customer_quick_ratio <- dfCvr$customer_quick_ratio/100
#
# # Join dfCvr on dfData
# dfData <- left_join(dfData,dfCvr,by="cvr")
#
# # Impute missing values in customer_equity_ratio and customer_quick_ratio with mean
# dfData$customer_equity_ratio[is.na(dfData$customer_equity_ratio)] <- mean(dfData$customer_equity_ratio, na.rm = T)
# dfData$customer_quick_ratio[is.na(dfData$customer_quick_ratio)] <- mean(dfData$customer_quick_ratio, na.rm = T)

dfOutstanding <- read_excel("./.AUX/Debitorer.xlsx")
dfOutstanding <- dfOutstanding %>%
  na.omit()
# Drop column 3
dfOutstanding <- dfOutstanding[,-3]
# Rename columns
colnames(dfOutstanding) <- c("date","department","outstanding","cvr")
# Set cvr as character
dfOutstanding$cvr <- as.character(dfOutstanding$cvr)
dfOutstanding$department <- as.factor(dfOutstanding$department)

dfData <- left_join(dfData,dfOutstanding,by=c("cvr"="cvr","department"="department","date"="date"))

dfData$outstanding[is.na(dfData$outstanding)] <- 0

# Replace NaN & Inf with NA
dfData <- replace(dfData, is.infinite(as.matrix(dfData)), NA)
dfData <- replace(dfData, is.nan(as.matrix(dfData)), NA)

# Replace NaN with NA
dfData <- replace(dfData, is.nan(as.matrix(dfData)), NA)
# Replace Inf with NA
dfData <- replace(dfData, is.infinite(as.matrix(dfData)), NA)

# Get the last observation of each job_no and determine if total contribution is positive or negative
dfDataContrib <- dfData %>%
  group_by(job_no) %>%
  mutate_at(colNum, cumsum) %>%
  filter(date == max(date))

# If contribution_cumsum is positive, set contribution_positive to 1, else set to 0
dfDataContrib$profitable <- ifelse(dfDataContrib$contribution_cumsum > 0, 1, 0)

# Omit all columns except job_no and profitable
dfDataContrib <- dfDataContrib[,c("job_no","profitable")]

# Join dfDataContrib on dfData
dfData <- left_join(dfData,dfDataContrib,by="job_no")

# Define risk as the log of the ratio between the realized and the S-curve. Omit NaN
dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(
    efficiency_risk = (costs_scurve_diff * costs_of_labor_share -(billable_rate_dep -0.90) + illness_rate_dep
                               + internal_rate_dep)^(2),
    overrun_risk = (costs_scurve_diff * costs_of_materials_share + costs_of_materials_share * billable_rate_dep
                                  + costs_of_materials_share)^(2)
  ) %>%
  do({
    if (any(is.na(.$contribution_scurve_diff)) || any(is.na(.$contribution_cumsum))) {
      data.frame(., risk = NA_real_)
    } else {
      model <- lm(contribution_scurve_diff ~ revenue_scurve_diff + costs_scurve_diff + 1 +
                                             billable_rate_dep + illness_rate_dep + internal_rate_dep
      , data = .)
      data.frame(., risk = exp(model$residuals))
    }
  })

# For each job get the total costs at the end of the job
dfData <- dfData %>%
  group_by(job_no) %>%
  arrange(date) %>%
  mutate(total_costs = costs_cumsum[length(costs_cumsum)],
         total_contribution = contribution_cumsum[length(contribution_cumsum)],
         total_margin = total_contribution/total_costs)


### Encode categorical variables ###
dfData <- dfData %>%
  mutate(wip = ifelse(status == "wip", 1, 0),
         dep_505 = ifelse(department == "505", 1, 0),
         posting_group_projekt = ifelse(job_posting_group == "PROJEKT", 1, 0))

# Drop status, department and job_posting_group
# dfData <- dfData[,-which(names(dfData) %in% c("status","department","job_posting_group"))]

# Responsible as factor
dfData$responsible <- as.factor(dfData$responsible)

# Adress as factor
dfData$address <- as.factor(dfData$address)

# Cvr as factor
dfData$cvr <- as.factor(dfData$cvr)

# Customer as factor
dfData$customer <- as.factor(dfData$customer)

# Job_no as factor
dfData$job_no <- as.factor(dfData$job_no)

## Encode description using word embeddings ##
# Create a dataframe with only the description and job_no
dfDesc <- dfData[,c('job_no','date','description')]
# Select only the latest description for each job_no
dfDesc <- dfDesc %>%
  group_by(job_no) %>%
  filter(date == max(date)) %>%
  ungroup()
dfDesc <- dfDesc[,c('job_no','description')]
# # Remove punctuation
# dfDesc$description <- gsub("[[:punct:]]", "", dfDesc$description)
# # Remove numbers
# dfDesc$description <- gsub("[[:digit:]]", "", dfDesc$description)
# # Remove extra white spaces
# dfDesc$description <- gsub("\\s+", " ", dfDesc$description)
# # Remove leading and trailing white spaces
# dfDesc$description <- trimws(dfDesc$description)
# Remove empty rows
dfDesc <- dfDesc[dfDesc$description != "",]
lText_Corpus <- Corpus(VectorSource(dfDesc$description)) # The corpus is a list object in R of type CORPUS

lText_Corpus <- tm_map(lText_Corpus, content_transformer(tolower))
lText_Corpus <- tm_map(lText_Corpus, removePunctuation)
lText_Corpus <- tm_map(lText_Corpus, removeNumbers)
lText_Corpus <- tm_map(lText_Corpus, removeWords, stopwords("da"))
lText_Corpus <- tm_map(lText_Corpus, stripWhitespace)
lText_Corpus <- tm_map(lText_Corpus, stemDocument,language = "danish")

lText_Corpus <- lText_Corpus[which(sapply(lText_Corpus, function(x) length(unlist(strsplit(as.character(x), " "))) > 0))]

#convert to document term matrix
new_docterm_corpus <- removeSparseTerms(DocumentTermMatrix(lText_Corpus),sparse = 0.975)

#find frequent terms
colS <- colSums(as.matrix(new_docterm_corpus))
doc_features <- data.table(name = attributes(colS)$names, count = colS)

#most frequent and least frequent words
doc_features[order(-count)][1:10] #top 10 most frequent words

ggplot(doc_features[count>5],aes(name, count)) +
  geom_bar(stat = "identity",fill=vColor[1],color='black') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "Words", y = "Count") +
  theme_elcon()
ggsave("./Results/Figures/0_description.pdf", width = 10, height = 5)
ggsave("./Results/Presentation/0_description.svg", width = 10, height = 5)

processed_data <- cbind(data.table(job_no = dfDesc$job_no),as.data.table(as.matrix(new_docterm_corpus)))
dfData <- left_join(dfData,processed_data,by="job_no")

# Remove description from dfData
dfData <- dfData[,-which(names(dfData) == "description")]

## Split test and train ##
# get list of unique job_no and sample 80% of the jobs
lJobNo <- unique(dfData$job_no)
lJobNoTrain <- sample(lJobNo,round(length(lJobNo)*0.8))
# Create a column in dfData with 1 if job_no is in lJobNoTrain, else 0
dfData$train <- ifelse(dfData$job_no %in% lJobNoTrain, 1, 0)

# Select last observation for each job_no
dfDataX <- dfData %>%
  group_by(job_no) %>%
  filter(date == max(date)) %>%
  ungroup()

# Get list of unique job_no where (sales_estimate_costs + production_estimate_costs + final_estimate_costs) == 0
# Calculate total costs
dfDataX <- dfDataX %>%
  mutate(estimate_costs_check = sales_estimate_costs + production_estimate_costs + final_estimate_costs)
lJobNoZero <- length(unique(dfDataX$job_no[dfDataX$estimate_costs_check == 0]))

# Drop jobs where budget_costs is 0
dfData <- dfData[!dfData$job_no %in% lJobNoZero,]

# Select last observation for each job_no
dfDataX <- dfData %>%
  group_by(job_no) %>%
  filter(date == max(date)) %>%
  ungroup()

# Number of jobs where production_estimate_costs is 0
nJobsZero <- length(unique(dfDataX$job_no[dfDataX$final_estimate_costs == 0]))

# Share of jobs where production_estimate_costs is 0
paste0("The share of jobs where production_estimate_costs is 0 is ",
       round(nJobsZero/length(unique(dfData$job_no))*100,2),
       "%.")

# List of jobs where final_estimate_costs is 0
lJobNoZero <- unique(dfDataX$job_no[dfDataX$final_estimate_costs == 0])

# If job_no is in lJobNoZero, set final_estimate_costs to budget_costs
dfData$final_estimate_costs[dfData$job_no %in% lJobNoZero] <- dfData$budget_costs[dfData$job_no %in% lJobNoZero]

### Assure Quality of Data ###
# Pick top 5 largest jobs by budget_costs
dfDataQA <- dfData %>%
  group_by(job_no) %>%
  filter(date == max(date)) %>%
  ungroup() %>%
  arrange((budget_costs)) %>%
  slice(1:5)

# Select job_no, date, budget_costs, sales_estimate_costs, production_estimate_costs, and final_estimate_costs
dfDataQA <- dfDataQA[,c("job_no","date","costs_cumsum","budget_costs","sales_estimate_costs","production_estimate_costs","final_estimate_costs")]