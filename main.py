# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Created on Mon Jan 30 23:02:18 2023

@author: Daan Gonzalez

Main code to clean, process and create predictions.
"""
import argparse
import time
import logging
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from houseprices_clean_data import fill_na_columns_with_value, \
                                   fill_all_missing_values
from houseprices_preprocessing_data import encode_fit, \
                                           encode_transform, \
                                           derived_features
from houseprices_predict_data import generate_predictions
from houseprices_eda_data import other_plots, plot_nulls_heatmap

timestr = time.strftime("%Y%m%d-%H%M%S")

drop_cols = ['Alley', 'PoolQC', 'MiscFeature', 'Fence', 'MoSold',
            'YrSold', 'MSSubClass','GarageType', 'GarageArea', 'GarageYrBlt',
            'GarageFinish','YearRemodAdd', 'LandSlope','BsmtUnfSF',
            'BsmtExposure','2ndFlrSF', 'LowQualFinSF', 'Condition1',
            'Condition2', 'Heating','Exterior1st', 'Exterior2nd',
            'HouseStyle', 'LotShape', 'LandContour', 'LotConfig',
            'Functional','BsmtFinSF1', 'BsmtFinSF2', 'FireplaceQu',
            'WoodDeckSF', 'GarageQual', 'GarageCond', 'OverallCond',
            'HeatingQC'
           ]

fill_cols = ["FireplaceQu", "BsmtQual", "BsmtCond",
             "BsmtFinType1", "BsmtFinType2"]

drop_final_cols = ['OverallQual', 'ExterCond', 'ExterQual', 'BsmtCond',
                   'BsmtQual','BsmtFinType1', 'BsmtFinType2',
                   'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                   'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                  ]



def config_logs():
    ''' Configures the logs results
    

    Returns
    logs.logs file
    -------
    None.

    '''
    try:
        logging.basicConfig(
            filename='logs/logs.log',
            level=logging.INFO,
            filemode='w',
            format='%(name)s - %(levelname)s - %(message)s')

    except RuntimeError:
        logging.error("No es posible configurar los logs")




def load_train_data(file_path):
    ''' Loads train file and creates a datafram
    

    Parameters
    ----------
    file_path : String path
        Path to load the datasets.

    Returns
    -------
    pd.Dataframe
        Corresponding dataframes.

    '''
    try:
        train = pd.read_csv(file_path+"train.csv")
        logging.info('Train shape: %s', str(train.shape))
        return train
    except FileNotFoundError:
        logging.error("El path del train set está mal.")
        print(f"El archivo train no está en este path: {file_path}")
        print("Busca en otro lugar, por lo pronto no podemos proceder.")





def load_test_data(file_path):
    ''' Loads test file and creates a dataframe
    

    Parameters
    ----------
    file_path : String path
        Path to load the datasets.

    Returns
    -------
    pd.Dataframe
        Corresponding dataframes.

    '''
    try:
        test = pd.read_csv(file_path+"test.csv")
        logging.info('Test shape: %s', str(test.shape))
        return test
    except FileNotFoundError:
        logging.error("El path del test set está mal.")
        print(f"El archivo test no está en este path: {file_path}")
        print("Busca en otro lugar, por lo pronto no podemos proceder.")


# an argument parser was defined to control variables
parser = argparse.ArgumentParser(
    prog="main",
    description="the maximum number of leafs in the random forest")

parser.add_argument('max_leaf', type=int)

if __name__ == "__main__":
    
    # arguments of this exe
    args = parser.parse_args()
    #Config logs
    config_logs()
    logging.info('The current timestamp is {}.'.format(timestr))
    
    #Load data
    train = load_train_data('')
    test = load_test_data('')
    
    # EDA plots
    plot_nulls_heatmap(train, timestr)
    other_plots(train, timestr)
    
    # Target
    y = train['SalePrice']
    
    # Fill NA
    train = fill_na_columns_with_value(train, fill_cols, "No")
    test = fill_na_columns_with_value(test, fill_cols, "No")
    
    # Fill all NA
    train = fill_all_missing_values(train)
    test = fill_all_missing_values(test)
    
    # Drop first useless variables
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)
    
    # Encode variables
    train_t = encode_fit(train)[0]
    transformer = encode_fit(train)[1]
    test_t = encode_transform(test, transformer)
    
    # Quick Shape before derived features
    print(train.shape)
    print(test.shape)
    print(train_t.shape)
    print(test_t.shape)
    
    # Derived some interaction features and drop useless variables
    train_t = derived_features(train_t)
    test_t = derived_features(test_t)
    train_t.drop(drop_final_cols, axis=1, inplace=True)
    test_t.drop(drop_final_cols, axis=1, inplace=True)
    
    # Train de model and score
    X = train_t
    
    model = RandomForestRegressor(max_leaf_nodes=args.max_leaf)
    model.fit(X, y)
    score = cross_val_score(model, X, y, cv=10)
    
    logging.info('The scikit-learn version is {}.'.format(sklearn.__version__))
    logging.info('Mean score from cross-validation: %s', str(score.mean()))
    
    try:
        # Get predictions
        generate_predictions(test_t['Id'], model, test_t, timestr)
    except OSError:
            logging.error("Directorio inválido o inexistente.")
