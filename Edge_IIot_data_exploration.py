import pandas as pd
import os
from source.helper_functions.save_outputs import save_output_to_file, save_plot_to_file
from source.helper_functions.visualization import  plot_attack_type_distribution, analyze_data_quality, plot_attack_label_distribution_edge
from io import StringIO
import matplotlib.pyplot as plt

cvs_dir = 'D:/dev/thesis/data/datasets/Edge-IIoTset/DNN-EdgeIIoT-dataset.csv'
save_dir = "D:/dev/thesis/data/data_exploration/data_exploration_EDGE_IIot"

# Load the dataset
data = pd.read_csv(cvs_dir)
# data info
data_info = data.info()
# Capture the output of data.info() into a string
buffer = StringIO()
data.info(buf=buffer)
data_info = buffer.getvalue()
# save outputs
save_output_to_file(data_info, 'data_info.txt', save_dir)

#save the first 5 rows of the dataset as csv
data_head = data.head()
data_head.to_csv(os.path.join(save_dir, 'data_head.csv'), index=False)

#save and plot data quality analysis
analyze_data_quality(data, save_dir)

#Colum info of Attack_label column, visualize counts in a pie chart
plot_attack_label_distribution_edge(data, 'Attack_label', save_dir)

#save and plot attack type distribution
plot_attack_type_distribution(data, 'Attack_type', save_dir)











#save and plot benign vs attack distribution
#plot_benign_vs_others(data, ' Label', save_dir)

#save and plot attack type distribution
#plot_attack_type_distribution(data, ' Label', save_dir)

#save and plot data quality analysis
#analyze_data_quality(data, save_dir)


