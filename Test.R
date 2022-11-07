################################################################################
#   Program to test the fit of ARCH(1), GARCH(1,1), e-GARCH(1,1),              #
#   and gjr-GARCH(1,1)                                                         #
#                                                                              #
#       Relies upon quantmod to collect returns of a list of tickers           #
#       Uses rugarch to specify and fit the different models to the returns    #
#                                                                              #
################################################################################

## install the packages
# install.packages("rugarch")
# install.packages("quantmod")

rm(list=ls()) 
setwd("/Users/tobiasbrammer/Library/Mobile Documents/com~apple~CloudDocs/Documents/Aarhus Uni/9. semester/Project Based Internship ")

# Load list of tickers
mData <- read.csv("igv.csv", sep = ",")

# mData <- na.omit(mData)

mData <- subset(mData, select = -c(Beskrivelse, Leveringsaddresse, Kommentar, Seneste.bogføringgsdato, Ansvarlig, Link.til.NAV))

mData <- na.omit(mData)

#vReg = lm(mData$DG.efter.justering ~ mData$Ressource..omkostning 
#                                          + mData$Vare..omkostning
#                                          + mData$Andre..omkostning
#                                          + mData$Beregnet.indtægt
#                                          + mData$Slut.vurdering.DG
#                                          + mData$Indregnet.DG.tidl..mdr..inkl...justering
#               , data=mData)

vReg = lm(mData$Beregnet.DB ~ mData$Ressource..omkostning 
          , data=mData)
summary(vReg)

any(is.na(mData$Beregnet.DB))
any(is.na(mData$Ressource..omkostning))

