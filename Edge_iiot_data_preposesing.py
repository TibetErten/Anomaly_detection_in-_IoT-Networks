import pandas as pd
import os
from source.helper_functions.save_outputs import save_output_to_file, save_plot_to_file
from source.helper_functions.visualization import  plot_attack_type_distribution, analyze_data_quality, plot_attack_label_distribution_edge
from io import StringIO
import matplotlib.pyplot as plt
from source.data_preposesing.Edge_IIot_dataprepossesing import DataPreprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


cvs_dir = 'D:/dev/thesis/data/datasets/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'
save_dir = 'D:/dev/thesis/data/datasets/Edge-IIoTset'
save_dir = "D:/dev/thesis/data/datasets/Edge-IIoTset/train,val and test"
preprossed_csv = "D:/dev/thesis/data/datasets/Edge-IIoTset/preprocessed_DNN.csv"

# Preprocess the dataset
#data = DataPreprocessing.preprocess_data(cvs_dir, save_dir)

# Load the dataset
data = pd.read_csv(preprossed_csv)
# check colum types and save outputs
data_info = data.info()
# Capture the output of data.info() into a string
buffer = StringIO()
data.info(buf=buffer)
data_info = buffer.getvalue()
# save outputs
save_output_to_file(data_info, 'data_info_preprossed.txt', save_dir)

# train and test data split using stratified sampling
X = data.drop(columns=['Attack_label', 'Attack_type'])
y = data['Attack_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
# Split the training data further into train (80%) and validation (20%) sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
# Save the split datasets
X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
X_val.to_csv(os.path.join(save_dir, 'X_val.csv'), index=False)
X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(save_dir, 'y_train.csv'), index=False)
y_val.to_csv(os.path.join(save_dir, 'y_val.csv'), index=False)
y_test.to_csv(os.path.join(save_dir, 'y_test.csv'), index=False)








