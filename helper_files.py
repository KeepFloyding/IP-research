from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Function for univariate linear regression
# Inputs
# df: dataframe of values
# x_array: key names for dataframe
# y_array: key names for dataframe
# Outputs
# df_out: dataframe of key results
# fig: multiple subplots 

def checkLinearFit(df,x_array,y_array,figsize=(10,10),n_cols=2,n_rows=2,alpha=0.2):
    
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
                X = df[item_x].values.reshape(-1,1)
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
    
    return df_out,fig

# Functions that help with classifiers


"""
General function to test any classifier
Inputs include:
    - X: values of a dataframe (should be scaled)
    - y: list of labels
    - clf_class: machine learning classifier
    - n_fold: Number of kfolds to iterate over
    - **kwargs: Number of additional parameters that are passed to the classifier
    
Outputs include:
    - y_pred: array of predicted values for X that are taken from the kfolds
    - df_score: dataframe containing accuracy, precision and recall for each Kfold iteration
"""


from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.model_selection import KFold, train_test_split

def test_classifier(X,y, clf_class,n_fold,**kwargs):
    
    # Construct a kfolds object
    kf = KFold(n_splits=n_fold,shuffle=True)
    y_checks = np.zeros(len(y))
    
    # Iterate through folds
    score = [];
    for train_index, test_index in kf.split(y):
        
        # Training classifier
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        
        # Predicting values and testing
        y_pred = clf.predict(X_test)
        score.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred),
                      recall_score(y_test,y_pred)])
        
        # Predicted values from cross-validation
        y_checks[test_index] = y_pred

    df_score = pd.DataFrame(score, columns=['Accuracy', 'Precision','Recall'])
    df_score.loc['mean'] = df_score.mean()
    
    return df_score, y_checks, clf
    
    
"""
Function that trains a random forest classifier and returns the feature importance
Inputs are:
    - X, y input variables
    - n_estimators: number of trees for random forest classifier
    - keys: feature labels 
"""

from sklearn.ensemble import RandomForestClassifier as RF

def return_feature_importance(X,y,keys,n_estimators = 100):

    # Using the random forest classifier, find out what are the main features that predict whether a user is likely to churn or not
    randomForest = RF(n_estimators)
    randomForest.fit(X,y)
    
    importances = randomForest.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]),keys[indices[f]])
    
