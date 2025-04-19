
import pandas as pd
import os
from source.helper_functions.save_outputs import save_output_to_file, save_plot_to_file
from source.helper_functions.visualization import  plot_benign_vs_others, plot_attack_type_distribution, analyze_data_quality
from io import StringIO

cvs_dir = 'D:/dev/thesis/data/CICD-2017_CVE'
save_dir = 'D:/dev/data/data_explorration'

# merging dataframes#

dfs = []

for dirname, _, filenames in os.walk(cvs_dir):
    for filename in filenames:
        dfs.append(pd.read_csv(os.path.join(dirname, filename)))

data = pd.concat(dfs, axis=0, ignore_index=True)


if False:  # Prevent this block from executing
    # save outputs

    data_info = data.info()

    # Capture the output of data.info() into a string

    buffer = StringIO()
    data.info(buf=buffer)
    data_info = buffer.getvalue()


    save_output_to_file(data_info, 'data_info.txt', save_dir)

#save and plot benign vs attack distribution
plot_benign_vs_others(data, ' Label', save_dir)

#save and plot attack type distribution
plot_attack_type_distribution(data, ' Label', save_dir)

#save and plot data quality analysis
analyze_data_quality(data, save_dir)

#save the merged data
data.to_csv(os.path.join(save_dir, 'merged_data.csv'), index=False)




















