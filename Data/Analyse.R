# update.packages(checkBuilt=TRUE, ask=FALSE)

library(dplyr)
library(tidyverse)
library(tidyr)
library(lubridate)
library(ggplot2)
library(ggthemes)
library(ggpubr)
library(feather)

# Get directory of file
dir <- "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
setwd(dir)
dfData <- feather::read_feather("dfData.feather")
# Set Dato as date
dfData$Dato <- as.Date(dfData$Dato, format = "%d-%m-%Y")

# Define color palette
col <- c('#006e64','#ffbb00','#c17150','#1e8c82','#734848','#dcfae9')

# Select random job number
set.seed(156342)
#sJobNo <- sample(dfData$Sagsnr.,1)
sJobNo <- 'S268074'
# Filter data with selected job number
dfSample <- dfData %>% filter(Sagsnr. == sJobNo)

# Order by date
dfSample <- dfSample %>% arrange(Dato)


# Create new column with the cumulative dækningsbidrag
dfSample <- dfSample %>% 
  mutate(CumDækningsbidrag = cumsum(Dækningsbidrag))
# Cumulative costs
dfSample <- dfSample %>% 
  mutate(CumOmkostninger = cumsum(Bogførte_omkostninger))
# Cumulative ressource usage
dfSample <- dfSample %>% 
  mutate(CumRessourceomkostninger = cumsum(Ressourceomkostninger))
# Cumulative vareomkostninger
dfSample <- dfSample %>% 
  mutate(CumVareomkostninger = cumsum(Vareomkostninger))

# Plot cumulative dækningsbidrag
ggplot(dfSample, aes(x = Dato, y = CumDækningsbidrag)) +
  geom_line() +
  labs(title = paste0("Kumulativt dækningsbidrag for ", dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       subtitle = "Dækningsbidrag = Faktureret indtægt - Bogført omkostning",
       x = "Dato",
       y = "Kumulativt dækningsbidrag") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red")

# Plot cumulative ressourceomkostninger
ggplot(dfSample, aes(x = Dato, y = CumRessourceomkostninger)) +
  geom_line() +
  labs(title = paste0("Kumulativt ressourceomkostninger for ", dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       subtitle = "Ressourceomkostninger = Løn + Omkostninger",
       x = "Dato",
       y = "Kumulativt ressourceomkostninger") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red")+ 
  scale_color_manual(values = col)


# Plot cumulative vareomkostninger
ggplot(dfSample, aes(x = Dato, y = CumVareomkostninger)) +
  geom_line() +
  labs(title = paste0("Kumulativt vareomkostninger for ", dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       subtitle = "Vareomkostninger = Vareforbrug + Omkostninger",
       x = "Dato",
       y = "Kumulativt vareomkostninger") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") + 
  scale_color_manual(values = col)

# Plot cumulative vareomkostninger and ressourceomkostninger
ggplot(dfSample, aes(x = Dato)) +
  geom_line(aes(y = CumRessourceomkostninger, color = "Ressourceomkostninger")) +
  geom_line(aes(y = CumVareomkostninger, color = "Vareomkostninger")) +
  labs(title = paste0("Ressourceomkostninger og vareomkostninger"),
       subtitle = paste0(dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       x = "Dato",
       y = "Kumulativt ressourceomkostninger og vareomkostninger") +
  # Format y-axis with thousands separator and decimal point
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))

### Theoretical s-curve ###
dBudget <- tail(dfSample$Slut_vurdering_omkostninger,1)
dBudgetRes <- tail(dfSample$Slut_vurdering_omkostninger,1) * sum(dfSample$Ressourceomkostninger)/sum(dfSample$Bogførte_omkostninger)
dBudgetVare <- tail(dfSample$Slut_vurdering_omkostninger,1) * sum(dfSample$Vareomkostninger)/sum(dfSample$Bogførte_omkostninger)
# Calculate the days between start and end of project
dStart <- as.numeric(head(dfSample$Dato,1))
dEnd <- as.numeric(tail(dfSample$Dato,1))
# Calculate the number of days between start and end of project
nDays <- as.numeric(dEnd - dStart)

# Iteratively calculate the theoretical s-curve for different values of dSkew and dPeak until the 
# theoretical s-curve fits the actual s-curve

# Define the function to calculate the sum of squared errors
SSE <- function(dSkew, dPeak) {
  # Calculate the theoretical s-curves according to \frac{1-\cos(\pi t^{s})+pt^{s}}{2+p}
  dfSample <- dfSample %>%
    mutate(Dato_numeric = as.numeric(Dato), # convert Dato to numeric
           TheoreticalScurveRes = dBudgetRes * (1-cos(pi*((Dato_numeric-dStart)/nDays)^(dSkew)) + dPeak *((Dato_numeric-dStart)/nDays)^(dSkew))/(2 + dPeak),
           TheoreticalScurveVare = dBudgetVare * (1-cos(pi*((Dato_numeric-dStart)/nDays)^(dSkew)) + dPeak *((Dato_numeric-dStart)/nDays)^(dSkew))/(2 + dPeak),
           TheoreticalScurve = dBudget * (1-cos(pi*((Dato_numeric-dStart)/nDays)^(dSkew)) + dPeak *((Dato_numeric-dStart)/nDays)^(dSkew))/(2 + dPeak)) # calculate theoretical s-curve
  
  # Calculate the sum of squared errors
  SSE <- sum((dfSample$TheoreticalScurveRes - dfSample$CumRessourceomkostninger)^2) + sum((dfSample$TheoreticalScurveVare - dfSample$CumVareomkostninger)^2)
  return(SSE)
}

# For each value of dSkew, calculate the sum of squared errors for different values of dPeak
dSkew <- seq(-1, 2, 0.01) # define the range of dSkew
dPeak <- seq(-1, 2, 0.01) # define the range of dPeak

# Initialize the matrix to store the sum of squared errors
SSEMatrix <- matrix(NA, nrow = length(dSkew), ncol = length(dPeak))

# Calculate the sum of squared errors for each combination of dSkew and dPeak
for (i in 1:length(dSkew)) {
  for (j in 1:length(dPeak)) {
    if (i <= nrow(SSEMatrix) && j <= ncol(SSEMatrix)) { # check if i and j are within bounds
      SSEMatrix[i,j] <- SSE(dSkew[i], dPeak[j])
    }
  }
}

# Find the minimum sum of squared errors and the corresponding values of dSkew and dPeak
minSSE <- min(SSEMatrix, na.rm = TRUE)
minSSEIndex <- which(SSEMatrix == minSSE, arr.ind = TRUE)
dSkewMinSSE <- dSkew[minSSEIndex[1]]
dPeakMinSSE <- dPeak[minSSEIndex[2]]

# Calculate the theoretical s-curves according to \frac{1-\cos(\pi t^{s})+pt^{s}}{2+p} with the values of dSkew and dPeak that minimize the sum of squared errors
dfSample <- dfSample %>%
  mutate(Dato_numeric = as.numeric(Dato), # convert Dato to numeric
         TheoreticalScurveRes = dBudgetRes * (1-cos(pi*((Dato_numeric-dStart)/nDays)^(dSkewMinSSE)) + dPeakMinSSE *((Dato_numeric-dStart)/nDays)^(dSkewMinSSE))/(2 + dPeakMinSSE),
         TheoreticalScurveVare = dBudgetVare * (1-cos(pi*((Dato_numeric-dStart)/nDays)^(dSkewMinSSE)) + dPeakMinSSE *((Dato_numeric-dStart)/nDays)^(dSkewMinSSE))/(2 + dPeakMinSSE),
         TheoreticalScurve = dBudget * (1-cos(pi*((Dato_numeric-dStart)/nDays)^(dSkewMinSSE)) + dPeakMinSSE *((Dato_numeric-dStart)/nDays)^(dSkewMinSSE))/(2 + dPeakMinSSE)) # calculate theoretical s-curve

# Plot the theoretical s-curves
# Varer
ggplot(dfSample, aes(x = Dato)) +
  geom_line(aes(y = TheoreticalScurveVare, color = "TheoreticalScurveVare")) +
  geom_line(aes(y = CumVareomkostninger, color = "CumVareomkostninger")) +
  geom_ribbon(aes(ymin = CumVareomkostninger, ymax = TheoreticalScurveVare), fill = "grey", alpha = 0.5) +
  labs(title = paste0("Theoretical s-curve for vareomkostninger"),
       subtitle = paste0(dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       x = "Dato",
       y = "Kumulative vareomkostninger") +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))

# Ressources 
ggplot(dfSample, aes(x = Dato)) +
  geom_line(aes(y = TheoreticalScurveRes, color = "TheoreticalScurveRes")) +
  geom_line(aes(y = CumRessourceomkostninger, color = "CumRessourceomkostninger")) +
  geom_ribbon(aes(ymin = CumRessourceomkostninger, ymax = TheoreticalScurveRes), fill = "grey", alpha = 0.5) +
  labs(title = paste0("Theoretical s-curve for ressourceomkostninger"),
       subtitle = paste0(dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       x = "Dato",
       y = "Kumulative ressourceomkostninger") +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))

# Total costs
ggplot(dfSample, aes(x = Dato)) +
  geom_line(aes(y = TheoreticalScurve, color = "TheoreticalScurveRes")) +
  geom_line(aes(y = CumOmkostninger, color = "CumRessourceomkostninger")) +
  geom_ribbon(aes(ymin = CumOmkostninger, ymax = TheoreticalScurve), fill = "grey", alpha = 0.5) +
  labs(title = paste0("Theoretical s-curve for omkostninger"),
       subtitle = paste0(dfSample$Sagsnr.,' - ',dfSample$Beskrivelse),
       x = "Dato",
       y = "Kumulative omkostninger") +
  scale_y_continuous(labels = scales::comma_format(big.mark = ".", decimal.mark = ",")) +
  theme_economist() +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  scale_color_manual(values = col) + 
  guides(color=guide_legend(title=NULL))
