# Helper functions to conduct statistical analysis

# ----------------------------------------------------------------------------------------------------
# Import the necesssary packages
# ----------------------------------------------------------------------------------------------------

# Data wrangling
import pandas as pd
import numpy as np

# Statistical tests
 from scipy.stats import ks_2samp

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plotting tools
import matplotlib.pyplot as plt


        
# ----------------------------------------------------------------------------------------------------
# Binning and sensitivity test
# ----------------------------------------------------------------------------------------------------

from sklearn.feature_selection import f_classif

# Bin the schools into seperate categories
def bin_groups(df, feature, bins, group_names):
    
    categories = pd.cut(df[feature],bins, labels=group_names)
    return categories