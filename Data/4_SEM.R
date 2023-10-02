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
library(cSEM)
library(reticulate)

# Source GetData
source('2_EDA.r')
rm(list=ls()[!grepl("dfData",ls())])
invisible(source('theme_elcon.R'))

use_python("C:\\Users\\tobr\\AppData\\Local\\Programs\\Python\\Python311")

py_run_string("from ESRNN import ESRNN")

from ESRNN.m4_data import *
from ESRNN.utils_evaluation import evaluate_prediction_owa
from ESRNN.utils_visualization import plot_grid_prediction

