# -*- coding: utf-8 -*-
# pylint: disable=no-member
"""
Created on Mon Jan 30 23:02:18 2023

@author: Daan Gonzalez

Main code to clean, process and create predictions.
"""
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

print('The scikit-learn version is {}.'.format(sklearn.__version__))

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

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# EDA plots
plot_nulls_heatmap(train)
other_plots(train)

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

model = RandomForestRegressor(max_leaf_nodes=250,)
model.fit(X, y)
score = cross_val_score(model, X, y, cv=10)
print(score.mean())

# Get predictions
generate_predictions(test_t['Id'], model, test_t)
