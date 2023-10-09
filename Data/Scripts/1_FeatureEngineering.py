import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import DanishStemmer
import re
from plot_config import *

# Set working directory
sDir = "C:/Users/tobr/OneDrive - NRGi A S/Projekter/ProjectBasedInternship/Data"
os.chdir(sDir)

# Read data
dfData = pd.read_parquet(f"{sDir}/dfData.parquet")
