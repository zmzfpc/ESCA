import pandas as pd
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--metrics_path', type=str, default=None, help='metric path')
args = parser.parse_args()

folder_path = args.metrics_path

if not os.path.exists(folder_path):
    print(f"Folder {folder_path} does not exist.")
else:
    # Get all the .csv files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    all_data = []
    # Loop through each file and read it into a DataFrame
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        all_data.append(df)

    assert len(all_data) == 7, "Number of rows in the combined DataFrame should be 7"
    all_data = pd.concat(all_data, ignore_index=True)
    # print(all_data.shape)
    

    output_file = os.path.join(folder_path, 'combined_metrics.csv')
    all_data.to_csv(output_file, index=False)

    print(f"Combined metrics saved to {output_file}")

    # Calculate the mean and standard deviation of each metric
    metrics = all_data.columns[1:]
    mean_std = {}   
    for metric in metrics:
        mean = all_data[metric].mean()
        std = all_data[metric].std()
        max_val = all_data[metric].max()
        min_val = all_data[metric].min()
        mean_std[metric] = (mean, std, max_val, min_val)
    # Create a DataFrame for mean and std   
    mean_std_df = pd.DataFrame(mean_std, index=['Mean', 'Std', 'Max', 'Min']).T
    mean_std_df.reset_index(inplace=True)
    mean_std_df.columns = ['Metric', 'Mean', 'Std', 'Max', 'Min']
    # Save the mean and std DataFrame to a CSV file
    mean_std_output_file = os.path.join(folder_path, 'mean_std_metrics.csv')
    mean_std_df.to_csv(mean_std_output_file, index=False)
    print(f"Mean and standard deviation of metrics saved to {mean_std_output_file}")



