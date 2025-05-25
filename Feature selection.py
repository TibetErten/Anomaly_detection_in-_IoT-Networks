import pandas as pd
import os
from source.feature_selection.XGBoost_feature_selection import analyze_feature_importance

#directories

save_dir = "D:/dev/thesis/data/feature_selection"
data_dir_train = "D:/dev/thesis/data/datasets/Edge-IIoTset/train,val and test/X_train.csv"
data_dir_test = "D:/dev/thesis/data/datasets/Edge-IIoTset/train,val and test/y_train.csv"

# Load the datasets
X_train = pd.read_csv(data_dir_train)
y_train = pd.read_csv(data_dir_test)

#turn bolean columns into integers in X_train
X_train = X_train.applymap(lambda x: 1 if x is True else (0 if x is False else x))
# check colum data types of xtrain
print("X_train data types:")
print(X_train.dtypes)






