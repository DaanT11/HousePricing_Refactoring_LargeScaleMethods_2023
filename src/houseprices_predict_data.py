# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 08:12:53 2023

@author: Daan Gonzalez

This script allows the user to make predictions and save them 
within a folder /results

This tool accepts a dataframe and a model artifact that is fitted in 
the main code,

This script requires that `pandas` be installed within the Python
environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * generate_predictions - returns a file with predicitions.

"""
import pandas as pd

def generate_predictions(identifier, model, dataframe):
    ''' Generates predictions based on pre-trained model
    

    Parameters:
    ----------
    identifier : (Column)
        Column with id
    model :(fitted model artifact)
        Model pre-trained
    dataframe : (pd.Dataframe)
        Dataframe to be used.

    Returns:
        A dataframe with corresponding id and prediction
    -------
    None.

    '''
    pred = model.predict(dataframe)
    submission = pd.DataFrame({
        "Id": identifier,
        "Predicted_SalePrice": pred
    })
    submission.to_csv("results/submission.csv", index=False)
