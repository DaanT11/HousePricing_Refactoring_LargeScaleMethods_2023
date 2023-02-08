# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 08:39:57 2023

@author: Daan Gonzalez

This script allows the user to plot and save within /plots folder
some EDA exploration graphs.

This tool accepts a dataframe to plot.

This script requires that seaborn and matplotlib, \
be installed within the Python environment you are running this script in.

This file can also be imported as a module and contains the following
functions:

    * plot_nulls_heatmap - plots heatmap of null variables
    * other_plots - plots like hist and violin plots
"""
import seaborn as sns
import matplotlib.pyplot as plt




def plot_nulls_heatmap(dataframe, time):
    ''' Creates a plot with null values for each column
    Parameters:
    ----------
    dataframe : (pd.Dataframe)
        Dataframe to be used.

    Returns:
        A plot .png format
    -------
    None.

    '''
    fig, axis = plt.subplots(figsize=(25, 10))
    sns.heatmap(data=dataframe.isnull(), yticklabels=False, ax=axis)
    fig.savefig("plots/plot_nulls_heatmap"+time+".png")




def other_plots(dataframe, time):
    ''' Generates some explorative plots
    Parameters:
    ----------
    dataframe : (pd.Dataframe)
        Dataframe to be used.

    Returns:
        A plot .png format
    -------
    None.

    '''
    fig, axis = plt.subplots(figsize=(25, 10))
    sns.countplot(x=dataframe['SaleCondition'])
    sns.histplot(x=dataframe['SaleType'], kde=True, ax=axis)
    sns.violinplot(x=dataframe['HouseStyle'], y=dataframe['SalePrice'], ax=axis)
    sns.scatterplot(x=dataframe["Foundation"], y=dataframe["SalePrice"],
                    palette='deep', ax=axis)
    plt.grid()
    fig.savefig("plots/other_plots"+time+".png")
