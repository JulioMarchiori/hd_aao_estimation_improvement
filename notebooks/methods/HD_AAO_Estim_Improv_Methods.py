#######################################################
#                                                     #
# Julio Cesar Marchiori Dias - s3095304               #
# Leiden University - BioInformatics Master Thesis    #
# LUMC BioSemantics Group                             #
# Improving the age at onset for Huntington's Disease #
#                                                     #
#######################################################

#####  IMPORTING RELEVANT LIBRARIES  #####

# Generic Libraries
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import os
from random import seed as rseed
import tensorflow as tf


# Models Libraries
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso, Lars, ElasticNet
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# Train and test Libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import RidgeCV

# Metrics Libraries
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score as r2_score_sk
from keras import backend as K
from sklearn.utils.class_weight import compute_sample_weight

# Neural Networks Libraries
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#################################################################


#####  FORMULA DEFINTIONS  #####

# Langbehn Formula
langbehn = lambda x: 21.54 + np.exp( 9.556 - (0.1460 * x))

#################################################################


#####  METHOD DEFINTIONS  #####



#################################################################
###  METRICS AND BASELINE METHODS  ###

def func(x, a, b, c):
    """ Internal Function used to calculate Langbehn formula
    """
    return a + np.exp(c - (b * x))


def refit_langbehn(subset):
    """ Function to calculate Langbehn Formula (refit)
    Parameters
    ----------
    subset: The HD dataframe from where 'cagigh' and 'hddiagn' will be collected for the AAO calculation

    Returns
    -------
    The estimated AAO  
    """
    a_langbehn = 21.54
    b_langbehn = 0.1460
    c_langbehn = 9.556
    
    x = subset['caghigh'].values
    y = subset['hddiagn'].values

    popt, pcov = curve_fit(func, x, y, p0=np.array([a_langbehn, b_langbehn, c_langbehn]), maxfev=5000)
    print('a={:.3f}\nb={:.3f}\nc={:.3f}'.format(*popt))
    return lambda x: func(x, *popt), popt


def refit_langbehn2(subset, target):
    """ Function to calculate Langbehn Formula (refit)
    Parameters
    ----------
    subset: The HD dataframe 
    target: The onset target (any motor, behaviour or cognitive one) 

    Both to be used for the AAO calculation with 'caghigh' information

    Returns
    -------
    The estimated AAO  
    """
    a_langbehn = 21.54
    b_langbehn = 0.1460
    c_langbehn = 9.556
    
    x = subset['caghigh'].values
    y = subset[target].values

    popt, pcov = curve_fit(func, x, y, p0=np.array([a_langbehn, b_langbehn, c_langbehn]), maxfev=5000)
    print('a={:.3f}\nb={:.3f}\nc={:.3f}'.format(*popt))
    return lambda x: func(x, *popt), popt


def evaluate(y_true, y_pred):
    """ Function to calculate the loss errors
    Parameters
    ----------
    y_true: Real Target 
    y_pred: Predicted Target

    Returns
    -------
    MAE, RMSE and R2  
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared = False)
    r2 = r2_score_sk(y_true, y_pred)
    return [mae, rmse, r2]


def calc_perc_improvement(old, new):
    """ Function to calculate gain between two different methods
    Parameters
    ----------
    old: Initial value collected 
    new: New value collected

    Returns
    -------
    Print the Calculataion. No return in values  
    """
    print((old - new).round(3))
    print('{:5.2f}%'.format(np.mean((old - new) / old) * 100))



#################################################################
###  CATEGORIZATION METHODS  ###

def cat_comorbid(row):
    """ Function to create a comorbid category based on 'mhbodsys' and 'mhstdy'
    Parameters
    ----------
    The entire patient row will be analyzed

    Returns
    -------
    1 if 'mhbodsys' is greater than zero and 'mhstdy' is less than zero.
    0 otherwise
    """
    if row['mhbodsys'] > 0.0 and row['mhstdy'] < 0:
        return 1
    else:
        return 0


def cat_nonpharmaco(row):  
    """ Function to create a nonpharmaceutic category based on 'cmtrt' and 'cmstdy'
    Parameters
    ----------
    The entire patient row will be analyzed

    Returns
    -------
    1 if 'cmtrt' is greater than zero and 'cmstdy' is less than zero.
    0 otherwise
    """
    if row['cmtrt'] > 0.0 and row['cmstdy'] < 0:
        return 1
    else:
        return 0
    
def cat_pharmaco(row):  
    """ Function to create a pharmaceutic category based on 'cmtrt__modify' and 'cmstdy_y'
    Parameters
    ----------
    The entire patient row will be analyzed

    Returns
    -------
    1 if 'cmtrt__modify' is different than string (NotUsedHere) and 'cmstdy_y' is less than zero.
    0 otherwise
    """
    if row['cmtrt__modify'] != 'NotUsedHere' and row['cmstdy_y'] < 0:
        return 1
    else:
        return 0

def cat_nutri(row):  
    """ Function to create a nutritional category based on 'cmtrt__modify_y' and 'cmstdy'
    Parameters
    ----------
    The entire patient row will be analyzed

    Returns
    -------
    1 if 'cmtrt__modify_y' is different than string (NotUsedHere) and 'cmstdy' is less than zero.
    0 otherwise
    """
    if row['cmtrt__modify_y'] != 'NotUsedHere' and row['cmstdy'] < 0:
        return 1
    else:
        return 0



#################################################################
###  PROCESSING DATA METHODS  ###

def OneHotEncodingFunc(df, target, num_columns, cat_columns, drop_cat):
    """ Function to execute the OneHot Encoding into categorical columns
    Parameters
    ----------
    df: The HD dataframe
    target: The feature target used into analysis
    num_columns: Numerical columns (these will not be affected)
    cat_columns: Categorical Columns (these will be encoded)
    drop_cat: Drop categorical columns (0 - No; 1 - Yes)

    Returns
    -------
    The new dataframe with the encoded columns,
    The list of the new encoded categorical columns
    """

    # Create subset to be one hot encoded
    oh_df = df[target + num_columns + cat_columns]
    oh_df = oh_df.reset_index()
    
    # One Hot Encoding
    oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cols = oh_encoder.fit_transform(oh_df[cat_columns])
    # Depending on scikit learn version, remove the comment of next line and comment the following one
#     new_cat_names = oh_encoder.get_feature_names(cat_columns)
    new_cat_names = oh_encoder.get_feature_names_out(cat_columns)    
    oh_df_enc = pd.DataFrame(encoded_cols, columns=new_cat_names)
    oh_df_enc = pd.concat([oh_df, oh_df_enc], axis=1)
    
    if drop_cat == True: 
        oh_df_enc = oh_df_enc.drop(columns=cat_columns)
    
    return oh_df_enc, new_cat_names.tolist()


def ScaleData(X_train, X_valid, y_train, y_valid, need_reshape):
    """ Function to Scale the data 
    Parameters
    ----------
    X_train: The input parameters to be used during training
    X_valid: The input parameters to be used during validation
    y_train: The target parameters to be used during training
    y_valid: The target parameters to be used during validation
    need_reshape: Flag to reshape the dataframe (0 - No; 1 - Yes)

    Returns
    -------
    The Scalar object created for target, 
    The Scalar object created for input, 
    The scaled X_train, 
    The scaled X_valid, 
    The scaled y_train, 
    The scaled y_valid, 
    The original target data 
    """

    y_orig = y_valid
    
    # Sandardization of data
    PredictorScalerTrain = StandardScaler()
    PredictorScalerTest = StandardScaler()
    TargetVarScalerTrain = StandardScaler()
    TargetVarScalerTest = StandardScaler()
    
    if need_reshape == True:
        y_train = y_train.reshape(-1, 1)
        y_valid = y_valid.reshape(-1, 1)

    # Storing the fit object for later reference
    PredictorScalerFitTrain = PredictorScalerTrain.fit(X_train)
    PredictorScalerFitTest = PredictorScalerTest.fit(X_valid)
    TargetVarScalerFitTrain = TargetVarScalerTrain.fit(y_train)
    TargetVarScalerFitTest = TargetVarScalerTest.fit(y_valid)

    # Generating the standardized values of X and y
    X_train = PredictorScalerFitTrain.transform(X_train)
    X_valid = PredictorScalerFitTest.transform(X_valid)
    y_train = TargetVarScalerFitTrain.transform(y_train)
    y_valid = TargetVarScalerFitTest.transform(y_valid)
    
    return TargetVarScalerFitTest, PredictorScalerFitTest, X_train, X_valid, y_train, y_valid, y_orig



#################################################################
###  FEATURE SELECTION METHODS  ###

def select_features(dataset, target_pp, columns_pp):
    """ Function to provide a feature selection based on 3 different methods:
    SelectKBest, Mutual Information and Lasso

    Parameters
    ----------
    dataset: The HD dataframe
    target_pp: The feature target used into analysis
    columns_pp: The input columns (to be ranked)

    Returns
    -------
    The Select KBest Results
    The Mutual Information Results
    The Lasso Results
    """

    ## SelectKBest ##
    
    # Segregating the Feature and Target
    A_df = dataset.fillna(value=0)
    X = A_df[columns_pp].values
    y = A_df[target_pp].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    scaleObj, _, X_train, X_test, y_train, y_test, y_orig = ScaleData(X_train, X_test, y_train, y_test,
                                                                           need_reshape=True)
    
    # Create results dataframe
    fs_results_k = pd.DataFrame(columns=['target', 'feature', 'score'])   
   
    # configure to select all features
    fs = SelectKBest(score_func = f_regression, k = 'all')
    
    # learn relationship from training data
    fs.fit(X_train, y_train)

    # what are scores for the features
    for i in range(len(fs.scores_)):
#         if fs.scores_[i] >= 0: #3
        fs_results_k.loc[len(fs_results_k)] = [target_pp, columns_pp[i], fs.scores_[i]]        

    fs_results_sk = fs_results_k.sort_values(['score'],ascending = [False])
        
    ## Information Gain ##
    
    # Apply Information Gain
    ig = mutual_info_regression(X_train, y_train)

    # Create a dictionary of feature importance scores
    feature_scores_i = {}
    for i in range(len(columns_pp)):
        feature_scores_i[columns_pp[i]] = ig[i]

    # Sort the features by importance score in descending order
    sorted_features_i = sorted(feature_scores_i.items(), key=lambda x: x[1], reverse=True)

    fs_results_i = pd.DataFrame(columns=['target', 'feature', 'score'])

    # Print the feature importance scores and the sorted features
    for feature_i, score_i in sorted_features_i:
        if score_i >= 0: #0.015
            fs_results_i.loc[len(fs_results_i)] = [target_pp, feature_i, score_i]

    fs_results_mi = fs_results_i
    
    ## Lasso ##
    
    # parameters to be tested on GridSearchCV
    params = {"alpha":np.arange(0.01, 10, 500)}

    # Number of Folds and adding the random state for replication
    kf=KFold(n_splits=5,shuffle=True, random_state=42)

    # Initializing the Model
    lasso = Lasso()

    # GridSearchCV with model, params and folds.
    lasso_cv=GridSearchCV(lasso, param_grid=params, cv=kf)
    lasso_cv.fit(X, y)
    
    # calling the model with the best parameter
    lasso1 = Lasso(alpha=lasso_cv.best_params_['alpha'])
    lasso1.fit(X_train, y_train)

    names = A_df[columns_pp].columns

    # Using np.abs() to make coefficients positive.  
    lasso1_coef = np.abs(lasso1.coef_)  
    
    feature_scores_l = {}
    for i in range(len(columns_pp)):
        feature_scores_l[columns_pp[i]] = lasso1_coef[i]   
    
    # Sort the features by importance score in descending order
    sorted_features_l = sorted(feature_scores_l.items(), key=lambda x: x[1], reverse=True)
    
    fs_results_l = pd.DataFrame(columns=['target', 'feature', 'score'])
    
    # Print the feature importance scores and the sorted features
    for feature_l, score_l in sorted_features_l:
        if score_l >= 0: #1.2
            fs_results_l.loc[len(fs_results_l)] = [target_pp, feature_l, score_l]

    fs_results_ls = fs_results_l
    
    
    return fs_results_sk, fs_results_mi, fs_results_ls



#################################################################
###  TUNING METHODS  ###


def tuneRandomForest(ml_dataset, cag_min, cag_max, columns, target, runOneHot):
    """ Function used to tune the Random Forest Model 
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    columns: The input columns used into analysis
    target: The feature target used into analysis
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The model tuning evaluation results
    """

    # Reset seed for each run
    seed_value = random.randrange(2 ** 32 - 1)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    rseed(seed_value)
        
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'bootstrap': [False, True],
        'max_depth': [90, 110, 150],
        'max_features': ['sqrt', 3, 15, 1.0],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [6],
        'n_estimators': [800, 1000, 1500]
    }
    # Create a based model
    rf = RandomForestRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 0)
         
    # Select data
    label = target
    cols=columns
    if runOneHot == 1:
        # Features Selected to be used
        num_columns_pp = cols[0]
        cat_columns_pp = cols[1]
        cols = num_columns_pp + cat_columns_pp
        drop_cat = True
    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    n = subset.shape[0]
    
    if runOneHot == 1:
        # Call One Hot Enconding Function
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    # Create Data    
    X_train, X_valid, y_train, y_valid = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)
    
    # Scale Data
    scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
    
    # Train Data
    print("Training Random Forest using GridSearchCV")
    grid_search.fit(X_train, y_train.ravel())   
    
    return grid_search


def tuneCatBoost(ml_dataset, cag_min, cag_max, columns, target, runOneHot):
    """ Function used to tune the Cat Boosting Model 
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    columns: The input columns used into analysis
    target: The feature target used into analysis
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The model tuning evaluation results
    """

    # Reset seed for each run
    seed_value = random.randrange(2 ** 32 - 1)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    rseed(seed_value)
        
    # Create the parameter grid based on the results of random search 
    param_grid = {'learning_rate': [0.005, 0.01, 0.05, 0.1],
                  'depth' : [3, 10, 20],
                  'iterations': [800, 1000, 1200],
                  'l2_leaf_reg': [1, 5, 9],
#                   'loss_function': ['MAE'],
#                   'bootstrap_type': ['Bayesian', 'Bernoulli'],
#                   'bagging_temperature': [0, 1, 5],
#                   'eval_metric': ['R2', 'MAE'],
#                   'min_data_in_leaf': [1, 3, 5],
#                   'max_leaves': [5, 10, 30]
                 }
    # Create a based model
    rf = CatBoostRegressor(silent=True)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 1)
          
    # Select data
    label = target
    cols=columns
    if runOneHot == 1:
        # Features Selected to be used
        num_columns_pp = cols[0]
        cat_columns_pp = cols[1]
        cols = num_columns_pp + cat_columns_pp
        drop_cat = True
    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    n = subset.shape[0]
    
    if runOneHot == 1:
        # Call One Hot Enconding Function
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    # Create Data    
    X_train, X_valid, y_train, y_valid = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)
    
    # Scale Data
    scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
    
    # Train Data
    print("Training Cat Boosting using GridSearchCV")
    grid_search.fit(X_train, y_train.ravel())
       
    return grid_search


def tuneAdaBoost(ml_dataset, cag_min, cag_max, columns, target, runOneHot):
    """ Function used to tune the Ada Boosting Model 
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    columns: The input columns used into analysis
    target: The feature target used into analysis
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The model tuning evaluation results
    """

    # Reset seed for each run
    seed_value = random.randrange(2 ** 32 - 1)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    rseed(seed_value)
        
    # Create the parameter grid based on the results of random search 
    param_grid = {'learning_rate': [0.03, 0.05, 0.1],
                  'loss' : ['linear', 'square'],
                  'n_estimators': [1000, 1500, 2000, 2500]
                 }
    # Create a based model
    rf = AdaBoostRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 0)
      
    # Select data
    label = target
    cols=columns
    if runOneHot == 1:
        # Features Selected to be used
        num_columns_pp = cols[0]
        cat_columns_pp = cols[1]
        cols = num_columns_pp + cat_columns_pp
        drop_cat = True
    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    n = subset.shape[0]
    
    if runOneHot == 1:
        # Call One Hot Enconding Function
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    # Create Data    
    X_train, X_valid, y_train, y_valid = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)
    
    # Scale Data
    scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
    
    # Train Data
    print("Training Ada Boosting using GridSearchCV")
    grid_search.fit(X_train, y_train.ravel())    
    
    return grid_search


def tuneGradBoost(ml_dataset, cag_min, cag_max, columns, target, runOneHot):
    """ Function used to tune the Gradient Boosting Model 
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    columns: The input columns used into analysis
    target: The feature target used into analysis
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The model tuning evaluation results
    """

    # Reset seed for each run
    seed_value = random.randrange(2 ** 32 - 1)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    rseed(seed_value)
        
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'max_depth': [3],
        'max_features': [15, 1.0],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [15],
        'n_estimators': [800],
        'learning_rate': [0.01, 0.03]
    }
    # Create a based model
    rf = GradientBoostingRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 5, n_jobs = -1, verbose = 0)
        
    # Select data
    label = target
    cols=columns
    if runOneHot == 1:
        # Features Selected to be used
        num_columns_pp = cols[0]
        cat_columns_pp = cols[1]
        cols = num_columns_pp + cat_columns_pp
        drop_cat = True
    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    n = subset.shape[0]
    
    if runOneHot == 1:
        # Call One Hot Enconding Function
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    # Create Data    
    X_train, X_valid, y_train, y_valid = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)
    
    # Scale Data
    scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
    
    # Train Data
    print("Training Gradient Boosting using GridSearchCV")
    grid_search.fit(X_train, y_train.ravel())   
    
    return grid_search


def tuneXGradBoost(ml_dataset, cag_min, cag_max, columns, target, runOneHot):
    """ Function used to tune the XGBoosting Model 
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    columns: The input columns used into analysis
    target: The feature target used into analysis
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The model tuning evaluation results
    """

    # Reset seed for each run
    seed_value = random.randrange(2 ** 32 - 1)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    rseed(seed_value)
        
    # Create the parameter grid based on the results of random search 
    param_grid = {
        'max_depth': [3],
        'max_features': [15, 1.0],
        'min_samples_leaf': [1, 2, 3],
        'min_samples_split': [15],
        'n_estimators': [800],
        'learning_rate': [0.01, 0.03]
    }
    # Create a based model
    rf = XGBRegressor()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 0)
        
    # Select data
    label = target
    cols=columns
    if runOneHot == 1:
        # Features Selected to be used
        num_columns_pp = cols[0]
        cat_columns_pp = cols[1]
        cols = num_columns_pp + cat_columns_pp
        drop_cat = True
    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    n = subset.shape[0]
    
    if runOneHot == 1:
        # Call One Hot Enconding Function
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    # Create Data    
    X_train, X_valid, y_train, y_valid = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)
    
    # Scale Data
    scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
    
    # Train Data
    print("Training XGBoosting using GridSearchCV")
    grid_search.fit(X_train, y_train.ravel())   
    
    return grid_search


def FindBestParams_NN(ml_dataset, ml_iter, columns, target, cag_min, cag_max, packages):
    """ Function used to tune the Neural Network Model 
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    ml_iter: Number of iterations to be used
    columns: The input columns used into analysis
    target: The feature target used into analysis
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    packages: Combo of batch size and epochs trial to be used

    Returns
    -------
    The model tuning evaluation results
    """

    # Prepare the data
    label = target
    num_columns_pp = columns[0]
    cat_columns_pp = columns[1]
    cols = num_columns_pp + cat_columns_pp
    drop_cat = True

    subset = ml_dataset.groupby('subjid').first().loc[:, label + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    
    subset, cat_cols = OneHotEncodingFunc(subset, label, num_columns_pp, cat_columns_pp, drop_cat)
    cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values  

      
    TrialNumber = 0   
    SearchResultsData = pd.DataFrame(columns=['TrialNumber', 'Parameters', 'MAE'])
    
    # Parameters definition
    units_list = [64, 128, 256]
    activation_list = ['relu'] #sigmoid
    hidden_layers = [2, 3, 4]
    learning_rate = [0.01, 0.005]
    epsilons = [0.001, 0.0001]    
    momentum = 0.8 
    
#     sgd = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False)     
#     optimizer = [ada] #, 'rmsprop', 'adam'
    
    for comb in packages:
        batch_size_trial = comb[0]
        epochs_trial = comb[1]      
    
    # initializing the trials
        for eps in epsilons:
            for lr in learning_rate:
                for hl in hidden_layers:
                    for act in activation_list: 
                        for unit_trial in units_list:            
                            TrialNumber+= 1
                            decay_rate = lr / epochs_trial
                            mae_tot = []
                            for i in range(ml_iter):
                                # Reset seed for each run
                                seed_value = random.randrange(2 ** 32 - 1)
                                np.random.seed(seed_value)
                                random.seed(seed_value)
                                os.environ['PYTHONHASHSEED'] = str(seed_value)
                                tf.random.set_seed(seed_value)
                                rseed(seed_value)
                                
                                # Create Optimizer    
                                ada = Adam(learning_rate=lr, epsilon=eps)
                                                              
                                # Split Data (train and test)   
                                X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)

                                # Scale Data
                                scaleObj, _, X_train, X_test, y_train, y_test, y_orig = ScaleData(X_train, X_test, y_train, y_test,
                                                                                                   need_reshape=True)  
                                # create ANN model
                                dim = X_train.shape[1]
                                model = Sequential()

                                # Defining the first layer of the model
                                model.add(Dense(units = unit_trial, input_dim = X_train.shape[1], kernel_initializer = 'normal', activation = act))

                                # Defining hidden layers
                                if hl==1:
                                    model.add(Dense(units = (unit_trial), kernel_initializer = 'normal', activation = act))
                                elif hl==2:
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act))
                                    model.add(Dense(units = (unit_trial), kernel_initializer = 'normal', activation = act))
                                elif hl==3:
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act))
                                    model.add(Dense(units = (3*unit_trial), kernel_initializer = 'normal', activation = act))                    
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act)) 
                                elif hl==4:
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act))
                                    model.add(Dense(units = (3*unit_trial), kernel_initializer = 'normal', activation = act))                    
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act))                     
                                    model.add(Dense(units = (unit_trial), kernel_initializer = 'normal', activation = act)) 
                                elif hl==5:
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act))
                                    model.add(Dense(units = (3*unit_trial), kernel_initializer = 'normal', activation = act))                    
                                    model.add(Dense(units = (3*unit_trial), kernel_initializer = 'normal', activation = act))                     
                                    model.add(Dense(units = (2*unit_trial), kernel_initializer = 'normal', activation = act))
                                    model.add(Dense(units = (unit_trial), kernel_initializer = 'normal', activation = act))

                                # Defining output layer
                                model.add(Dense(1, kernel_initializer = 'normal'))

                                # Compiling the model
                                model.compile(loss = 'mean_absolute_error', optimizer = ada)

                                # Fitting the ANN to the Training set
                                model.fit(X_train, y_train, batch_size = batch_size_trial, epochs = epochs_trial, verbose = 0)

                                # Evaluating the model
                                y_pred = model.predict(X_test)
                                y_new_inverse = scaleObj.inverse_transform(y_pred)

                                # Reset session
#                                 model.reset_states()
                                K.clear_session()

                                # R2 Value
#                                 r2 = r2_score_sk(y_test, y_pred)
                                r2 = r2_score_sk(y_orig, y_new_inverse)
                                mae = mean_absolute_error(y_orig, y_new_inverse)
                                mse = mean_squared_error(y_orig, y_new_inverse, squared = True)
                                rmse = mean_squared_error(y_orig, y_new_inverse, squared = False)
                                mae_tot.append(mae)

                                # printing the results of the current iteration
                                print("Run:",TrialNumber,'-',i,'/ batch:', batch_size_trial,'-', 'epochs:',epochs_trial,
                                      '-', 'Losses (mae/mse/rmse/r2):', "%.3f" %mae, "%.3f" %mse, "%.3f" %rmse, "%.3f" %r2,'-Units:', unit_trial, '-HL:', hl, '-Act:', act, '-LR:', lr, '-Eps:', eps)

                            SearchResultsData.loc[len(SearchResultsData)] = [TrialNumber, str(lr)+'-'+str(eps)+'-'+str(act)+'-'+str(batch_size_trial)+'-'+str(epochs_trial)+'-'+str(unit_trial)+'-'+str(hl), np.mean(mae_tot)]

    return(SearchResultsData)



#################################################################
###  BUILDING MODEL METHODS  ###

def train_models(ml_dataset, cag_min, cag_max, columns, target, NNparams, runOneHot):
    """ Function used to train different models:
    (Langbehn, RandomForest, CatBoosting, GradientBooting, AdaBoosting and NeuralNetwork)
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    columns: The input columns used into analysis
    target: The feature target used into analysis
    NNparams: Combo of batch size and epochs trial to be used
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The average summary with the results
    """

    print((ml_dataset[(ml_dataset['caghigh'] >=cag_min) & (ml_dataset['caghigh'] <= cag_max)].copy()).shape)
    langbehn_refitted, _ = refit_langbehn2(ml_dataset[(ml_dataset['caghigh'] >=cag_min) & (ml_dataset['caghigh'] <= cag_max)].copy(), target=target)
    langbehn_models = [langbehn, langbehn_refitted]
    langbehn_names = ['Langbehn original', 'Langbehn refitted']
    
    models = langbehn_models + [feedforwardNNFunc, 
                                RandomForestRegressor(max_depth=150, min_samples_split=6, n_estimators=1000),
                                CatBoostRegressor(depth=5, iterations=800, learning_rate=0.03, silent=True),
                                GradientBoostingRegressor(learning_rate= 0.03, max_depth= 3, min_samples_leaf= 3, 
                                                          min_samples_split= 15, max_features=15, n_estimators= 800),
                                AdaBoostRegressor(n_estimators=2500, learning_rate=0.03, loss='square'),
                               ]
    
    names = langbehn_names + ['FeedForwardNN',
                              'Random Forest',
                              'CatBoost',
                              'GradientBoosting',
                              'AdaBoosting',
                             ]
    all_results = []
    
    # Evaluation    
    for name, mod in zip(names, models):
        print('-' * 40)
        print(name)
        if 'Langbehn' in name:
            results = train(ml_dataset, mod, cag_range=(cag_min, cag_max), fit=0, cols=['caghigh'], target=target,
                             NNparams=NNparams, runOneHot=0)
        elif name == 'FeedForwardNN':
            results = train(ml_dataset, mod, cag_range=(cag_min, cag_max), fit=1, cols=columns, target=target,
                             NNparams=NNparams, runOneHot=1)
        elif name == 'Linear Regression' and runOneHot==1:
            columns_lr = columns[0]+columns[1]
            results = train(ml_dataset, mod, cag_range=(cag_min, cag_max), fit=2, cols=columns_lr, target=target,
                             NNparams=NNparams, runOneHot=0)                
        else:
            results = train(ml_dataset, mod, cag_range=(cag_min, cag_max), fit=2, cols=columns, target=target,
                             NNparams=NNparams, runOneHot=runOneHot)
        results.index = [name]
        all_results.append(list(results.reset_index().values.reshape(-1)))
        print()
    
    # Save
    summary = pd.DataFrame(all_results, columns=['Model', 'MAE', 'RMSE', 'R2'])\
                .set_index('Model')\
                .sort_values('MAE', ascending=True)
    summary.to_csv(os.path.join('tables', 'summary_AAO_{}-{}_models.csv'.format(cag_min, cag_max)), float_format='%.3f')
    return summary


def train(train_dataset, model, cag_range, fit, cols, target, NNparams, runOneHot):
    """ Function used to train different models, invoked by 'train_models' Method:
    (Langbehn, RandomForest, CatBoosting, GradientBooting, AdaBoosting and NeuralNetwork)
    
    Parameters
    ----------
    train_dataset: The HD dataframe
    model: name of the Model to be trained
    cag_range: combo of 'caghigh' range to to be filtered
    fit: category to execute the fit method of each model (0 - Langbehn; 1 - NeuralNetwork; 2 - Other Models)
    cols: Combo of the input columns used into analysis (Numerical and Categorical)
    target: The feature target used into analysis
    NNparams: Combo of batch size and epochs trial to be used
    runOneHot: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)

    Returns
    -------
    The average summary with the results (MAE, RMSE and R2)
    """

    # Select data
    label = target
    if runOneHot == 1:
        # Features Selected to be used
        num_columns_pp = cols[0]
        cat_columns_pp = cols[1]
        cols = num_columns_pp + cat_columns_pp
        drop_cat = True
    subset = train_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_range[0]) & (subset['caghigh'] <= cag_range[1]))]
    print('Dropped {} samples'.format(subset.isnull().any(axis=1).sum()))
    subset.dropna(inplace=True)
    n = subset.shape[0]
    print('{} samples left'.format(n)) 
    
    if runOneHot == 1:

        # Call One Hot Enconding Function
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols

    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    # Evaluation results
    test_results = []
    
    # Fold labels and predictions
    test_labels = []
    test_predictions = []
    
    # Train and Evaluate
    for i in range(10):
        
        # Reset seed for each run
        seed_value = random.randrange(2 ** 32 - 1)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        tf.random.set_seed(seed_value)
        rseed(seed_value)
 
        # Use KFold 
        kf = KFold(n_splits=5, shuffle=True, random_state=seed_value)
        for fold, (train_idx, test_idx) in enumerate(kf.split(input_data, targets)):
            # K-fold
            X_train, X_valid = input_data[train_idx], input_data[test_idx]
            y_train, y_valid = targets[train_idx], targets[test_idx]

            # Scale, 
            if fit == 2:
                scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
                model.fit(X_train, y_train)
                y_valid = y_orig
            elif fit == 1:
                model = feedforwardNNFunc(X_train.shape[1])
                scaleObj, _, X_train, X_valid, y_train, y_valid, y_orig = ScaleData(X_train, X_valid, y_train, y_valid, need_reshape=True)
                model.fit(X_train, y_train, batch_size = NNparams[0], epochs = NNparams[1], verbose = 0)
                y_valid = y_orig
            # prediction
            try:
                pred = model.predict(X_valid)
                if fit == 1:
                    y_pred = scaleObj.inverse_transform(pred)
                    pred = y_pred
                elif fit == 2:
                    pred = pred.reshape(-1, 1)
                    y_pred = scaleObj.inverse_transform(pred)
                    pred = y_pred                 
            except:
                pred = model(X_valid.reshape(-1))

            # Evaluate
            test_results.append(evaluate(y_valid, pred))

    return pd.DataFrame(np.mean(test_results, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])


def buildFinalModel(ml_model, ml_dataset, ml_iter, columns, target, cag_min, cag_max, NNparams, scale_flag, valid_df='null'):
    """ Function used to build the final models (using independent random seed):
    (RandomForest, CatBoosting, GradientBooting, AdaBoosting and NeuralNetwork)
    
    Parameters
    ----------
    ml_model: Model Name. Possible values: 'RandomForest', 'CatBoost','GradientBoosting','AdaBoost' and 'FFNN'
    ml_dataset: The HD dataframe
    ml_iter: Number of iterations to be used
    columns: Combo of the input columns used into analysis (Numerical and Categorical)
    target: The feature target used into analysis
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    NNparams: Combo of batch size and epochs trial to be used
    scale_flag: Flag to determine if data needs to be scaled (0 - No; 1 - Yes)
    valid_df: Validation Test dataframe

    Returns
    -------
    The average summary with the results (MAE, RMSE and R2)
    """

    # Evaluation results
    test_results = []
    fit = 2
    
    # Prepare the data
    label = target
    num_columns_pp = columns[0]
    cat_columns_pp = columns[1]
    cols = num_columns_pp + cat_columns_pp
    drop_cat = True

    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
    cols = num_columns_pp + cat_cols
    
    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    
    if valid_df != 'null':
        valid_subset = valid_df.groupby('subjid').first().loc[:, [label] + cols]
        valid_subset = valid_subset.loc[((valid_subset['caghigh'] >= cag_min) & (valid_subset['caghigh'] <= cag_max))]
        valid_subset.dropna(inplace=True)    
        valid_subset, cat_valid_cols = OneHotEncodingFunc(valid_subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols_valid = num_columns_pp + cat_valid_cols
        # Get input and labels
        X_valid = valid_subset[cols_valid].values
        y_valid = valid_subset[label].values

    for i in range(ml_iter):
    
        # Reset seed for each run
        seed_value = random.randrange(2 ** 32 - 1)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        tf.random.set_seed(seed_value)
        rseed(seed_value)       

        # Split Data (train and test)   
#         X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value, 
#                                                            stratify = subset['caghigh'])
        
        X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)

        # Scale Data
        if scale_flag == 1:
            scaleObj, _, X_train, X_test, y_train, y_test, y_orig = ScaleData(X_train, X_test, y_train, y_test, need_reshape=True)
        
        # Create a based model
        if ml_model == 'RandomForest':
            model = RandomForestRegressor(max_depth=150, min_samples_split=6, n_estimators=1000, random_state=seed_value)           
        elif ml_model == 'CatBoost':
            model = CatBoostRegressor(depth=3, iterations=800, learning_rate=0.01, l2_leaf_reg=1, silent=True, random_state=seed_value)
        elif ml_model == 'GradientBoosting':
            model = GradientBoostingRegressor(learning_rate= 0.03, max_depth= 5, n_estimators= 800, random_state=seed_value)           
        elif ml_model == 'AdaBoost':
            model = AdaBoostRegressor(n_estimators=2500, learning_rate=0.03, loss='square', random_state=seed_value)          
        elif ml_model == 'FFNN':
            model = feedforwardNNFunc(X_train.shape[1], NNparams[2])
        else:
            return print("Invalid Model")

        # Train Data
        if ml_model == 'FFNN':
            model.fit(X_train, y_train, batch_size = NNparams[0], epochs = NNparams[1], verbose = 0)
        else:
            model.fit(X_train, y_train.ravel())
        try:
            pred = model.predict(X_test)
            if scale_flag == 1:
                y_test = y_orig
                if fit == 1:
                    y_pred = scaleObj.inverse_transform(pred)
                    pred = y_pred
                elif fit == 2:
                    pred = pred.reshape(-1, 1)
                    y_pred = scaleObj.inverse_transform(pred)
                    pred = y_pred                 
        except:
            pred = model(X_test.reshape(-1))

        # Evaluate       
        if valid_df != 'null': 
            pred_valid = model.predict(X_valid)
            print("Validation Results", evaluate(y_valid, pred_valid))
   
        test_results.append(evaluate(y_test, pred))

    return pd.DataFrame(np.mean(test_results, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2']), test_results                              
                                

def tryAllModels(ml_dataset, ml_iter, columns, target, cag_min, cag_max, NNparams):
    """ Function used to build the final models (sharing the same random seed):
    (RandomForest, CatBoosting, GradientBooting, AdaBoosting and NeuralNetwork)
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    ml_iter: Number of iterations to be used
    columns: Combo of the input columns used into analysis (Numerical and Categorical)
    target: The feature target used into analysis
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    NNparams: Combo of batch size and epochs trial to be used

    Returns
    -------
    The average summary with the results (MAE, RMSE and R2) and the history results, per model, where:
        RandomForest - rf and test_results_rf
        CatBoosting - cb and test_results_cb
        GradientBooting - gb and test_results_gb
        AdaBoosting - ab and test_results_ab
        NeuralNetwork - nn and test_results_nn
    """

    # Evaluation results
    test_results_rf = []
    test_results_ab = []
    test_results_gb = []
    test_results_cb = []
    test_results_nn = []
    
    # Prepare the data
    label = target
    num_columns_pp = columns[0]
    cat_columns_pp = columns[1]
    cols = num_columns_pp + cat_columns_pp
    drop_cat = True

    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
    cols = num_columns_pp + cat_cols
    
    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values

    for i in range(ml_iter):
    
        # Reset seed for each run
        seed_value = random.randrange(2 ** 32 - 1)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        tf.random.set_seed(seed_value)
        rseed(seed_value)       
        print("Execution no", i+1, "using seed value", seed_value)
        
        # Split Data (train and test)   
        X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)

        # Scale Data
        scaleObj, _, X_train, X_test, y_train, y_test, y_orig = ScaleData(X_train, X_test, y_train, y_test, need_reshape=True)

        # Create a based models
        model_rf = RandomForestRegressor(max_depth=150, min_samples_split=6, n_estimators=1000, random_state=seed_value)                     
        model_cb = CatBoostRegressor(depth=3, iterations=800, learning_rate=0.01, l2_leaf_reg=1, silent=True, random_state=seed_value)
        model_gb = GradientBoostingRegressor(learning_rate= 0.03, max_depth= 5, n_estimators= 800, random_state=seed_value)
        model_ab = AdaBoostRegressor(n_estimators=2500, learning_rate=0.03, loss='square', random_state=seed_value)          
        model_nn = feedforwardNNFunc(X_train.shape[1], NNparams[2])

        # Train the models
        model_nn.fit(X_train, y_train, batch_size = NNparams[0], epochs = NNparams[1], verbose = 0)
        model_rf.fit(X_train, y_train.ravel())
        model_cb.fit(X_train, y_train.ravel())
        model_gb.fit(X_train, y_train.ravel())
        model_ab.fit(X_train, y_train.ravel())       
            
        y_test = y_orig
        
        try:
            pred_rf = model_rf.predict(X_test)
            pred_cb = model_cb.predict(X_test)
            pred_gb = model_gb.predict(X_test)
            pred_ab = model_ab.predict(X_test)
            pred_nn = model_nn.predict(X_test)
            
            pred_rf = pred_rf.reshape(-1, 1)
            pred_cb = pred_cb.reshape(-1, 1)
            pred_gb = pred_gb.reshape(-1, 1)
            pred_ab = pred_ab.reshape(-1, 1)

            y_pred_rf = scaleObj.inverse_transform(pred_rf)
            pred_rf = y_pred_rf     
            y_pred_cb = scaleObj.inverse_transform(pred_cb)
            pred_cb = y_pred_cb    
            y_pred_gb = scaleObj.inverse_transform(pred_gb)
            pred_gb = y_pred_gb    
            y_pred_ab = scaleObj.inverse_transform(pred_ab)
            pred_ab = y_pred_ab 

            y_pred_nn = scaleObj.inverse_transform(pred_nn)
            pred_nn = y_pred_nn    
                
        except:
            print("ERROR During Prediction")

        test_results_rf.append(evaluate(y_test, pred_rf))
        test_results_gb.append(evaluate(y_test, pred_gb))
        test_results_ab.append(evaluate(y_test, pred_ab))
        test_results_cb.append(evaluate(y_test, pred_cb))
        test_results_nn.append(evaluate(y_test, pred_nn))
        
        rf = pd.DataFrame(np.mean(test_results_rf, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
        gb = pd.DataFrame(np.mean(test_results_gb, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
        ab = pd.DataFrame(np.mean(test_results_ab, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
        cb = pd.DataFrame(np.mean(test_results_cb, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
        nn = pd.DataFrame(np.mean(test_results_nn, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])

    return rf, gb, ab, cb, nn, test_results_rf, test_results_gb, test_results_ab, test_results_cb, test_results_nn


def tryTwoModels(ml_dataset, ml_iter, columns, target, cag_min, cag_max, NNparams, oneHotenc, valid_df):
    """ Function used to build the two final models (sharing the same random seed):
    (CatBoosting and NeuralNetwork)
    
    Parameters
    ----------
    ml_dataset: The HD dataframe
    ml_iter: Number of iterations to be used
    columns: Combo of the input columns used into analysis (Numerical and Categorical)
    target: The feature target used into analysis
    cag_min: Minimum value of 'caghigh' to be filtered
    cag_max: Maximum value of 'caghigh' to be filtered
    NNparams: Combo of batch size and epochs trial to be used
    oneHotenc: Flag to execute the OneHot Encoding (0 - No; 1 - Yes)
    valid_df: Validation Test dataframe 

    Returns
    -------
    The average summary with the results (MAE, RMSE and R2), the history results and validation results, where:
        CatBoosting - cb, test_results_cb, val_cb and valid_results_cb
        NeuralNetwork - nn, test_results_nn, val_nn and valid_results_nn
    The seed list used for each execution - seed_values_list
    Langbehn result obtained over the validation dataframe - lb_eval
    """
  
    # Evaluation results
    test_results_cb = []
    test_results_nn = []
    valid_results_cb = []
    valid_results_nn = []
    seed_values_list = []
    val_nn = []
    val_cb = []
    
    # Prepare the data
    lb_done = 0
    label = target
    num_columns_pp = columns[0]
    cat_columns_pp = columns[1]
    cols = num_columns_pp + cat_columns_pp   

    subset = ml_dataset.groupby('subjid').first().loc[:, [label] + cols]
    subset = subset.loc[((subset['caghigh'] >= cag_min) & (subset['caghigh'] <= cag_max))]
    subset.dropna(inplace=True)
    if oneHotenc == 1:
        drop_cat = True
        subset, cat_cols = OneHotEncodingFunc(subset, [label], num_columns_pp, cat_columns_pp, drop_cat)
        cols = num_columns_pp + cat_cols
    
    if valid_df.empty == False:
        cols_valid = columns[0] + columns[1]
        valid_subset = valid_df.groupby('subjid').first().loc[:, [label] + cols_valid]
        valid_subset = valid_subset.loc[((valid_subset['caghigh'] >= cag_min) & (valid_subset['caghigh'] <= cag_max))]
        valid_subset.dropna(inplace=True)
        
        # Langbehn
        lb_df_x = valid_subset['caghigh'].values
        lb_df_y = valid_subset[label].values
        lb_df_y_pred = [langbehn(x_i) for x_i in lb_df_x]
        lb_eval = evaluate(lb_df_y, lb_df_y_pred)
        lb_done = 1
        
        if oneHotenc == 1:
            drop_cat = True
            valid_subset, cat_valid_cols = OneHotEncodingFunc(valid_subset, [label], columns[0], columns[1], drop_cat)
            cols_valid = num_columns_pp + cat_valid_cols

        # Get input and labels
        X_valid = valid_subset[cols_valid].values
        y_valid = valid_subset[label].values        
    
    # Langbehn
    if lb_done == 0:
        lb_df_x = subset['caghigh'].values
        lb_df_y = subset[label].values
        lb_df_y_pred = [langbehn(x_i) for x_i in lb_df_x]
        lb_eval = evaluate(lb_df_y, lb_df_y_pred) 
    
    # Get input and labels
    input_data = subset[cols].values
    targets = subset[label].values
    print("Total Patients:", len(targets))

    for i in range(ml_iter):
    
        # Reset seed for each run
        seed_value = random.randrange(2 ** 32 - 1)
        np.random.seed(seed_value)
        random.seed(seed_value)
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        tf.random.set_seed(seed_value)
        rseed(seed_value)       
   
        # Split Data (train and test)   
        X_train, X_test, y_train, y_test = train_test_split(input_data, targets, test_size=0.20, random_state=seed_value)
       
        # Scale Data
        scaleObj, _, X_train, X_test, y_train, y_test, y_orig = ScaleData(X_train, X_test, y_train, y_test, need_reshape=True)

        # Create a based models                   
        model_cb = CatBoostRegressor(depth=3, iterations=1000, learning_rate=0.01, l2_leaf_reg=1, silent=True, random_state=seed_value)     
        model_nn = feedforwardNNFunc(X_train.shape[1], NNparams[2])

        # Train the models
        model_nn.fit(X_train, y_train, batch_size = NNparams[0], epochs = NNparams[1], verbose = 0)
        model_cb.fit(X_train, y_train.ravel())
               
        y_test = y_orig
        
        try:
            pred_cb = model_cb.predict(X_test)
            pred_nn = model_nn.predict(X_test)
            
            pred_cb = pred_cb.reshape(-1, 1)
            y_pred_cb = scaleObj.inverse_transform(pred_cb)
            pred_cb = y_pred_cb    

            y_pred_nn = scaleObj.inverse_transform(pred_nn)
            pred_nn = y_pred_nn    
                
        except:
            print("ERROR During Prediction")
          
        test_results_cb.append(evaluate(y_test, pred_cb))
        test_results_nn.append(evaluate(y_test, pred_nn))
        seed_values_list.append(seed_value)
               
        if valid_df.empty == False:
            scaleObjY, scaleObjX, X_valid_new, _, y_valid_new, _, y_valid_orig = ScaleData(X_valid, X_valid, y_valid, y_valid, need_reshape=True)
            pred_valid = model_cb.predict(X_valid_new)
            pred_valid = pred_valid.reshape(-1, 1)
            y_pred_valid = scaleObjY.inverse_transform(pred_valid)
            pred_valid = y_pred_valid
            valid_results_cb.append(evaluate(y_valid_orig, pred_valid))
            
            pred_valid_nn = model_nn.predict(X_valid_new)
            y_pred_valid_nn = scaleObjY.inverse_transform(pred_valid_nn)
            pred_valid_nn = y_pred_valid_nn
            valid_results_nn.append(evaluate(y_valid_orig, pred_valid_nn))

    cb = pd.DataFrame(np.mean(test_results_cb, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
    nn = pd.DataFrame(np.mean(test_results_nn, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
   
    if valid_df.empty == False:
        val_nn = pd.DataFrame(np.mean(valid_results_nn, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
        val_cb = pd.DataFrame(np.mean(valid_results_cb, axis=0).reshape(1, -1), columns=['MAE', 'RMSE', 'R2'])
            
    return cb, nn, test_results_cb, test_results_nn, seed_values_list, lb_eval, val_cb, val_nn, valid_results_cb, valid_results_nn
                              

def feedforwardNNFunc(dim, hl):
    """ Function used to build the Neural Network model. Invoked by other building methods.
    
    Parameters
    ----------
    dim: Dimension of input layer
    hl: number of hidden layers. Possible values: 2 or 3.

    Returns
    -------
    The built model
    """

    K.clear_session()

    # Prepare Neural Network Parameters
    if hl == 2:       
        units_1 = 128
        units_2 = 256
        units_3 = 128
    elif hl == 3:
        units_1 = 64
        units_2 = 128
        units_3 = 256
        units_4 = 128

    verb = 0
    act = 'relu'
    initializer = 'normal'
    optimizer = 'adam'
    lr = 0.005   #0.005
    eps = 0.001 #0.0001
    
    ada = Adam(learning_rate=lr, epsilon=eps)
    
    # create ANN model
    model = Sequential()

    # Defining the input layer
    model.add(Dense(units = units_1, input_dim = dim, kernel_initializer = initializer, activation = act))
    
    # Defining the hidden layers   
    model.add(Dense(units = units_2, kernel_initializer = initializer, activation = act))
    model.add(Dense(units = units_3, kernel_initializer = initializer, activation = act))
    if hl == 3: model.add(Dense(units = units_4, kernel_initializer = initializer, activation = act))

    # Since we will be predicting a single number (age)
    model.add(Dense(1, kernel_initializer = initializer))

    # Compiling the model
    model.compile(loss='mean_absolute_error', optimizer=ada, metrics=['mean_absolute_error'])

    return model

#################################################################
