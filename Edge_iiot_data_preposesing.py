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

# Train and test data split using stratified sampling
X = data.drop(columns=['Attack_label', 'Attack_type'])
y_multiclass = data['Attack_type']
y_binary = data['Attack_label']

# First split for test set
X_train, X_test, y_train_multi, y_test_multi, y_train_binary, y_test_binary = train_test_split(
    X, y_multiclass, y_binary,
    test_size=0.2, 
    stratify=y_multiclass,
    random_state=42
)

# Split training data into train and validation
X_train, X_val, y_train_multi, y_val_multi, y_train_binary, y_val_binary = train_test_split(
    X_train, y_train_multi, y_train_binary,
    test_size=0.2,
    stratify=y_train_multi,
    random_state=42
)

# Save the split datasets
# Features
X_train.to_csv(os.path.join(save_dir, 'X_train.csv'), index=False)
X_val.to_csv(os.path.join(save_dir, 'X_val.csv'), index=False)
X_test.to_csv(os.path.join(save_dir, 'X_test.csv'), index=False)

# Multiclass labels (Attack_type)
y_train_multi.to_csv(os.path.join(save_dir, 'y_train_type.csv'), index=False)
y_val_multi.to_csv(os.path.join(save_dir, 'y_val_type.csv'), index=False)
y_test_multi.to_csv(os.path.join(save_dir, 'y_test_type.csv'), index=False)

# Binary labels (Attack_label)
y_train_binary.to_csv(os.path.join(save_dir, 'y_train_label.csv'), index=False)
y_val_binary.to_csv(os.path.join(save_dir, 'y_val_label.csv'), index=False)
y_test_binary.to_csv(os.path.join(save_dir, 'y_test_label.csv'), index=False)







