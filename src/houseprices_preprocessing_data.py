# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 08:12:53 2023

@author: Daan Gonzalez

This script allows the user to preprocess the features and create new ones.

This tool accepts a dataframe to fit pipeline and columns to derive new 
interactions.

This script requires that `numpy` and `sklearn`, be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * encode_fit - fits the pipeline
    * encode_transform - transforms using the endode_fit artifact second returns
    * derived_features - derives some features
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

ordinal_cols = ['BsmtQual', 'BsmtCond', 'ExterQual', 'ExterCond', 'KitchenQual',
            'PavedDrive', 'Electrical', 'BsmtFinType1', 'BsmtFinType2',
            'Utilities', 'MSZoning', 'Foundation', 'Neighborhood', 'MasVnrType',
            'SaleCondition', 'RoofStyle', 'RoofMatl', 'BldgType', 'SaleType',
            'Street', 'CentralAir']

numeric_cols = ['Id', '1stFlrSF','3SsnPorch','BedroomAbvGr','BsmtFullBath','BsmtHalfBath',
                'EnclosedPorch','Fireplaces','FullBath','GarageCars','GrLivArea',
                'HalfBath','KitchenAbvGr','LotArea','LotFrontage',
                'MasVnrArea','MiscVal','OpenPorchSF','OverallQual',
                'PoolArea','ScreenPorch','TotRmsAbvGrd',
                'TotalBsmtSF','YearBuilt']

cat_pipeline = Pipeline(steps=[
    ('ordinal_encoder', OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value = np.nan))
     ])
     
num_pipeline = Pipeline(steps=[
    ('num_encoder', SimpleImputer(missing_values=np.nan, strategy='mean'))
     ])
     
# Create ColumnTransformer to apply pipeline for each column type
cols_transform = ColumnTransformer(transformers=[
    ('num_pipeline', num_pipeline, numeric_cols),
    ('cat_pipeline', cat_pipeline, ordinal_cols)
], remainder='drop', n_jobs=-1, verbose_feature_names_out=False,).set_output(transform="pandas")




def encode_fit(train_dataframe):
    '''Encode a fixed list of columns, already determined in a train dataset
    

    Parameters:
    ----------
    train dataframe : (pd.Dataframe)
        Dataframe to be used to fit the encoder.

    Returns:
        A dataframe with transfornation and column transformer artifact
    -------
    None.

    '''
    new_train_dataframe = train_dataframe.copy(deep=True)
    
    new_train_dataframe = cols_transform.fit_transform(new_train_dataframe)

    return new_train_dataframe, cols_transform




def encode_transform(test_dataframe, cols_transform):
    '''Encode a fixed list of columns, already determined in a train dataset
    

    Parameters:
    ----------
    test dataframe : (pd.Dataframe)
        Dataframe to be used to apply the fitted encoder.

    Returns:
        A dataframe with corresponding columns encoded
    -------
    None.

    '''
    new_test_dataframe = test_dataframe.copy(deep=True)
    
    new_test_dataframe = cols_transform.transform(new_test_dataframe)

    return new_test_dataframe




def derived_features(dataframe):
    '''Calculates derived features based on interactions or other calculus.
    

    Parameters:
    ----------
    dataframe : (pd.Dataframe)
        Dataframe to be used.

    Returns:
        A dataframe with corresponding derived features.
    -------
    None.

    '''
    new_dataframe = dataframe.copy(deep=True)
    
    new_dataframe['BsmtRating'] = new_dataframe['BsmtCond'] * new_dataframe['BsmtQual']
    new_dataframe['ExterRating'] = new_dataframe['ExterCond'] * new_dataframe['ExterQual']
    new_dataframe['BsmtFinTypeRating'] = new_dataframe['BsmtFinType1'] * new_dataframe['BsmtFinType2']
    
    new_dataframe['BsmtBath'] = new_dataframe['BsmtFullBath'] + new_dataframe['BsmtHalfBath']
    new_dataframe['Bath'] = new_dataframe['FullBath'] + new_dataframe['HalfBath']
    new_dataframe['PorchArea'] = new_dataframe['OpenPorchSF'] + \
                                 new_dataframe['EnclosedPorch'] + \
                                 new_dataframe['3SsnPorch'] + \
                                 new_dataframe['ScreenPorch']
    return new_dataframe
