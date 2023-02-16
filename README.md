# HousePricing_Refactoring_LargeScaleMethods_2023
This is a refactoring exercise from the house pricing prediction code, based on coding good practices and PEP8 standards

## Brief Summary
Repository to refactor code based on ITAM MSc in Data Science assignement from Large Scale Methods course.

## Requirements
All scrpts are based on python langaguage greater than 3.1 version, the only strict requirement is to have at least 1.2.0 scklearn version.

Review the environments.yaml file to use with conda. Use the following command to re-create the current project conda environment:
conda env create --file environments.yaml

## Data 
Data can be downloaded from Kaggle: www.kaggle.com

## RepoStructure
- data: files from train and test in csv format.
- plots: plots from eda
- msc: empty
- results: predictions in csv format
- src: Four .py scripts to explore, clean, process and predict.
- logs: has the log file to catch errors and info.

## Usage
Main .py file will call the other modules and just be sure to have files in the same folder.


