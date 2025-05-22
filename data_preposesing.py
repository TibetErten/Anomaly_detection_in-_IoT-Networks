
import pandas as pd
import numpy as np
import os

data_dir = 'D:/dev/data/data_explorration/merged_data.csv'

#load the merged data
df = pd.read_csv(data_dir)

#remove spaces from column names
df.columns = df.columns.str.replace(' ', '')

#remove duplicates
df = df.drop_duplicates()

#remove rows with NAN values
df = df.dropna()

# Reset the indexes of the dataframe
df.reset_index(drop=True, inplace=True)

#show categorical colums
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(cat_cols)

#print column names
print(df.columns)











