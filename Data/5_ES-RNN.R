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
library(signal)
library(tsfeatures)

# Source GetData
source('1_FeatureEngineering.r')

# Source theme_elcon
invisible(source('theme_elcon.R'))

rm(list=ls()[!grepl("dfData",ls())])

# Select random job no from dfData
sJobNo <- 'S333620'

sJobNo <- sample(unique(dfData$job_no),1)
# Create dfDataJob
dfDataJob <- dfData[dfData$job_no == sJobNo,]

# Create a dataframe with date and numeric variables
colNum <- names(dfDataJob)[sapply(dfDataJob, is.numeric)]
dfDataJobNum <- dfDataJob[,c('date',colNum)]

# Run fast fourier transform on numeric variables
dfDataJobNumFFT <- dfDataJobNum %>%
  mutate(across(where(is.numeric), fft))

# Plot fft of numeric variables
par(mfrow = c(2,1))
plot(dfDataJobNumFFT$date,Mod(dfDataJobNumFFT$revenue_scurve_diff), type = 'l', main = 'FFT of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,ifft(dfDataJobNumFFT$revenue_scurve_diff), type = 'l', main = 'Inverse FFT of Revenue S-Curve Difference')

# Forecast the last 20% of the data
# Create a dataframe with date and numeric variables
dfDataJobNumFFT$forecast <- 0
dfDataJobNumFFT$forecastES <- 0
# Forecast with ARMA
library(forecast)
iH <- floor(nrow(dfDataJobNumFFT)*0.2)
## Forecast with ARMA
# Fill first 80% of forecast with real data
dfDataJobNumFFT$forecast[1:(nrow(dfDataJobNumFFT)-iH)] <- Mod(dfDataJobNumFFT$revenue_scurve_diff)[1:(nrow(dfDataJobNumFFT)-iH)]
# Forecast last 20% of data with ARMA
dfDataJobNumFFT$forecast[(nrow(dfDataJobNumFFT)-iH+1):nrow(dfDataJobNumFFT)] <- forecast(auto.arima(Mod(dfDataJobNumFFT$revenue_scurve_diff[1:(nrow(dfDataJobNumFFT)-iH)])),h = iH)$mean[1:iH]


dfDataJobNumFFT$forecastES[1:(nrow(dfDataJobNumFFT)-iH)] <- Mod(dfDataJobNumFFT$revenue_scurve_diff)[1:(nrow(dfDataJobNumFFT)-iH)]
# Forecast last 20% of data with ETS
dfDataJobNumFFT$forecastES[(nrow(dfDataJobNumFFT)-iH+1):nrow(dfDataJobNumFFT)] <- forecast(ets(Mod(dfDataJobNumFFT$revenue_scurve_diff[1:(nrow(dfDataJobNumFFT)-iH)])),h = iH)$mean[1:iH]

par(mfrow = c(5,1))
title <- paste0('ARMA Forecast of FFT Revenue S-Curve Difference for Job No: ',sJobNo)
plot(dfDataJobNumFFT$date,Mod(dfDataJobNumFFT$revenue_scurve_diff), type = 'l', main = 'FFT of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,ifft(dfDataJobNumFFT$revenue_scurve_diff), type = 'l', main = 'Inverse FFT of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,dfDataJobNumFFT$forecast, type = 'l', main = 'Forecast of FFT Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,ifft(dfDataJobNumFFT$forecast), type = 'l', main = 'Inverse Forecast of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,(ifft(dfDataJobNumFFT$revenue_scurve_diff) - ifft(dfDataJobNumFFT$forecast))^2, type = 'l', main = 'Squared Forecast Error (ARMA)')

par(mfrow = c(5,1))
title <- paste0('ETS Forecast of FFT Revenue S-Curve Difference for Job No: ',sJobNo)
plot(dfDataJobNumFFT$date,Mod(dfDataJobNumFFT$revenue_scurve_diff), type = 'l', main = 'FFT of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,ifft(dfDataJobNumFFT$revenue_scurve_diff), type = 'l', main = 'Inverse FFT of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,dfDataJobNumFFT$forecastES, type = 'l', main = 'Forecast of FFT Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,ifft(dfDataJobNumFFT$forecastES), type = 'l', main = 'Inverse Forecast of Revenue S-Curve Difference')
plot(dfDataJobNumFFT$date,(ifft(dfDataJobNumFFT$revenue_scurve_diff) - ifft(dfDataJobNumFFT$forecastES))^2, type = 'l', main = 'Squared Forecast Error (ETS)')

# Difference between forecasts
par(mfrow = c(1,1))
plot(dfDataJobNumFFT$date,dfDataJobNumFFT$forecast - dfDataJobNumFFT$forecastES, type = 'l', main = 'Difference between ARMA and ETS forecast')

dfDataJobNum$date[1]

dAlpha <- holt_parameters(dfDataJobNum$revenue_scurve_diff)[1]
dBeta <- holt_parameters(dfDataJobNum$revenue_scurve_diff)[2]

ts <- ts(dfDataJobNum$revenue_scurve_diff, start = dfDataJobNum$date[1], end = dfDataJobNum$date[nrow(dfDataJobNum)])
plot(ts)
s_t <- dfDataJobNum$revenue_scurve_diff*dBeta



heterogeneity(dfDataJobNum$revenue_scurve_diff)
trev_num(dfDataJobNum$revenue_scurve_diff)