# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:49:08 2023

@author: Daan Gonzalez

This script allows the user to clean the data.

This tool accepts comma separated value files (.csv).

This file can also be imported as a module and contains the following
functions:

    * fill_na_columns_with_value - fill na's with specific value
    * fill_all_missing_values - fill all na's with fixed rule
"""

# houseprices_clean_data.py

def fill_na_columns_with_value(dataframe, columns, na_value):
    ''' Imputes NA values in the columns with specific value
    

    Parameters:
    ----------
    dataframe : (pd.Dataframe)
        Dataframe to be used.
    columns : (List)
        List of characters with the column names.
    na_value :( String or Number)
        Value to be imputed for na.

    Returns:
        A dataframe with corresponding columns imputed
    -------
    None.

    '''
    
    new_dataframe = dataframe.copy(deep=True)
    
    for col in columns:
        new_dataframe[col].fillna(na_value, inplace=True)
        
    return new_dataframe



def fill_all_missing_values(dataframe):
    ''' Imputes all columns with corresponding mode and mode based on data type
    

    Parameters:
    ----------
    dataframe : (pd.Dataframe)
        Dataframe to be used.

    Returns:
        A imputed dataframe with corresponding mean and mode columns
     -------
    None.

    '''
    
    new_dataframe = dataframe.copy(deep=True)
    
    for col in new_dataframe.columns:
        if new_dataframe[col].dtype in ['float', 'int']:
            new_dataframe[col].fillna(new_dataframe[col].mean(), inplace=True)
        else:
            new_dataframe[col].fillna(new_dataframe[col].mode()[0], inplace=True)
            
    return new_dataframe