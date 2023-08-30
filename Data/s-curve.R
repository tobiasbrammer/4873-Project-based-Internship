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

# Plot different configuration of s-curves (1/(1+exp(-kx))^a.
# k = 10, a = 2
# k = 10, a = 3
# k = 10, a = 4
# k = 10, a = 5
# k = 10, a = 6
# k = 10, a = 7

k <- 6
a <- c(1,2,3,4,5,6,7,8,9,10)

# Create data frame with date and progress
dfScurve <- data.frame(date = seq(as.Date("2019-01-01"), as.Date("2021-12-01"), by = "month"))

# Calculate progress
dfScurve <- dfScurve %>%
  mutate(days_since_start = date - min(date),
         total_days = max(dfScurve$date) - min(dfScurve$date), # make total_days a difftime object
         progress = as.numeric(days_since_start) / as.numeric(total_days))

# Calculate s-curves
dfScurve <- dfScurve %>%
  mutate(scurve_1 = (1/(1+exp(-k*(progress-0.5))))^1,
         scurve_2 = (1/(1+exp(-k*(progress-0.5))))^1.1,
         scurve_3 = (1/(1+exp(-k*(progress-0.5))))^1.2,
         scurve_4 = (1/(1+exp(-k*(progress-0.5))))^1.3,
         scurve_5 = (1/(1+exp(-k*(progress-0.5))))^1.4,
         scurve_6 = (1/(1+exp(-k*(progress-0.5))))^1.5,
         scurve_7 = (1/(1+exp(-k*(progress-0.5))))^1.6,
         scurve_8 = (1/(1+exp(-k*(progress-0.5))))^1.7,
         scurve_9 = (1/(1+exp(-k*(progress-0.5))))^1.8,
         scurve_10 = (1/(1+exp(-k*(progress-0.5))))^1.9)

# Plot s-curves
ggplot(dfScurve, aes(x = date)) +
  geom_line(aes(y = scurve_1), color = "cornflowerblue") +
  geom_line(aes(y = scurve_2), color = "red") +
  geom_line(aes(y = scurve_3), color = "blue") +
  geom_line(aes(y = scurve_4), color = "green") +
  geom_line(aes(y = scurve_5), color = "orange") +
  geom_line(aes(y = scurve_6), color = "purple") +
  geom_line(aes(y = scurve_7), color = "black") +
  geom_line(aes(y = scurve_8), color = "brown") +
  geom_line(aes(y = scurve_9), color = "pink") +
  geom_line(aes(y = scurve_10), color = "yellow") +
  theme_elcon() +
  labs(title = "S-curves with different a-values",
       subtitle = paste0("k = ",k),
       x = "Date",
       y = "Progress",
       caption = "Source: ELCON A/S") +
  scale_y_continuous(breaks = seq(0,1,0.1))