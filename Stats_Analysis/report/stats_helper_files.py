# Helper functions to conduct statistical analysis

# ----------------------------------------------------------------------------------------------------
# Import the necesssary packages
# ----------------------------------------------------------------------------------------------------

# Data wrangling
import pandas as pd
import numpy as np

# Statistical tests
from sklearn.feature_selection import f_classif

# Regressors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Plotting tools
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------
# Binning and statistical tests
# ----------------------------------------------------------------------------------------------------

def bin_groups(df, feature, bins, group_names):
    
    categories = pd.cut(df[feature],bins, labels=group_names)
    return categories

def plotHistograms(df, feature_list,nrows, ncols, figsize=(20,10), group_1 = 'Inactive', group_2 = 'Active'):
    
    plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize);

    count = 1;
    for feature in feature_list:
        
        plt.subplot(nrows,ncols,count)
        plt.hist(df[df['categories']==group_1][feature])
        plt.hist(df[df['categories']==group_2][feature],alpha=0.8)
        plt.xlabel(feature)
        count += 1

def non_parametric_test(df_test, feature_list, test_type):

    score_array = [];
    pval_array = [];
    
    for feature in feature_list:
        test_x = df_test[df_test['categories']=='Active'][feature]
        test_y = df_test[df_test['categories']=='Inactive'][feature]

        score, pval = test_type(test_x,test_y)
        
        score_array.append(score)
        pval_array.append(pval)
        
    df_report = pd.DataFrame({'Feature':feature_list,'Score':score_array,'P val':pval_array})
    
    return df_report.sort_values('P val')
    
# ----------------------------------------------------------------------------------------------------
# Sensitivity test for binning
# ----------------------------------------------------------------------------------------------------

def sensitivity_on_bin(df, feature_to_bin, features_to_evaluate, bins, group_names, cut_off_array, test_type = 'ANOVA'):

    store= [];
    
    group_0 = group_names[0]
    group_1 = group_names[1]
    group_2 = group_names[2]

    for item in cut_off_array:

        # Updating bins
        bins[2] = item
        
        # Binning the groups
        df_test = df
        df_test['categories'] = bin_groups(df_test,feature_to_bin, bins,group_names)
        num_active = df_test['categories'].value_counts().loc[group_2]
        df_test = df_test[df_test['categories']!= group_1]

        # Determining difference in means between active and inactive groups
        mu_active_inactive = df_test[df_test['categories']==group_2][features_to_evaluate].mean() - df_test[df_test['categories']==group_0][features_to_evaluate].mean()

        # Performing an ANOVA test
        if test_type == 'ANOVA':
            X = df_test[features_to_evaluate]
            y = [1 if item == group_2 else 0 for item in df_test['categories']]
            F, pval = f_classif(X, y)
            df_score = pd.DataFrame({'Key':X.keys(),'Score':F,'p values':pval,'Cut off':np.ones(len(X.keys()))*item, 'Num active':num_active, 'Group_2 - Group_0':mu_active_inactive[X.keys()]})

        # Performing non-parametric statistical tests
        else:
            X = df_test[features_to_evaluate]
            df_report = non_parametric_test(df_test, features_to_evaluate, test_type)

            #df_score = pd.DataFrame({'Key':X.keys(),'p values':df_report['P val'],'Group_2 - Group_0':mu_active_inactive[X.keys()].values})
            df_score = pd.DataFrame({'Key':X.keys(),'Score':df_report['Score'],'p values':df_report['P val'],'Cut off':np.ones(len(features_to_evaluate))*item, 'Num active':num_active, 'Group_2 - Group_0':mu_active_inactive[X.keys()].values})

        # Storing the array as a Dataframe
        store.append(df_score)

    df_score = pd.concat(store)
    
    return df_score

def plot_sensitivity(df_score, x_range, y_range):

    n_rows = len(x_range)
    n_cols = len(y_range)
    plt.subplots(n_rows,n_cols, figsize=(10,40))

    count = 1
    for item_x in x_range:
        
        df_test = df_score[df_score['Key'] == item_x]
        for item_y in y_range:

            plt.subplot(n_rows,n_cols,count)
            plt.scatter(df_test['Cut off'],df_test[item_y])
            plt.xlabel('Cut off')
            plt.ylabel(item_y)
            plt.title(item_x)
            
            count += 1


# ----------------------------------------------------------------------------------------------------
# Regression Analysis
# ----------------------------------------------------------------------------------------------------

# Function for univariate linear regression
# Inputs
# df: dataframe of values
# x_array: key names for dataframe
# y_array: key names for dataframe
# Outputs
# df_out: dataframe of key results
# fig: multiple subplots 

def checkLinearFit(df,x_array,y_array,figsize=(10,10),n_cols=2,n_rows=2,alpha=0.2,log=[0,0]):
    
    # Creating storage arrays
    x_choose = []
    y_choose = []
    coef_array = [];
    intercept_array = [];
    R_array = []
    
    plt.subplots(n_rows,n_cols,figsize=figsize)
    
    # Cycling through each dependent variable
    count = 1;
    for item_x in x_array:
        for item_y in y_array:
            
            if item_y != item_x:

                # Grabbing X and Y values
                
                if log[0]=='yes':
                    X = np.log(df[item_x].values.reshape(-1,1) + 1)
                else:
                    X = df[item_x].values.reshape(-1,1)
                
                if log[1] == 'yes':
                    Y = np.log(df[item_y].values.reshape(-1,1) + 1)
                else: 
                    Y = df[item_y].values.reshape(-1,1)

                # Training the model
                clf = LinearRegression()
                clf.fit(X,Y)
                Y_pred = clf.predict(X)

                # Storing important values in a dataframe
                x_choose.append(item_x)
                y_choose.append(item_y)
                coef_array.append(clf.coef_[0][0])
                intercept_array.append(clf.intercept_[0])
                R_array.append(r2_score(Y, Y_pred))

                # Plotting results
                plt.subplot(n_rows,n_cols,count)
                plt.scatter(X,Y,alpha=alpha)
                plt.xlabel(item_x)
                plt.ylabel(item_y)
                plt.plot(X,Y_pred)
                count += 1

    # Storing results in a dataframe
    df_out = pd.DataFrame({'X':x_choose,'Y':y_choose,'Coef':coef_array,'intercept':intercept_array,'R^2':R_array})
    
    return df_out
