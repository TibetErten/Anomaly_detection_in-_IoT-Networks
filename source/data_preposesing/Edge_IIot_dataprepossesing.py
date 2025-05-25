import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
from source.helper_functions.save_outputs import save_output_to_file


class DataPreprocessing:
    """
    A class containing methods for preprocessing the Edge-IIoT dataset.
    All methods are static and can be used without instantiating the class.
    """
    
    @staticmethod
    def load_dataset(data_path):
        """
        Load the dataset from CSV file.
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("Loading dataset...")
        return pd.read_csv(data_path, low_memory=False)

    @staticmethod
    def get_data_info(data):
        """
        Get information about the dataset's shape and columns.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            str: Information about the dataset
        """
        info = f"Dataset shape: {data.shape}\n"
        info += f"Columns: {', '.join(data.columns)}\n"
        return info

    @staticmethod
    def remove_unwanted_columns(data):
        """
        Remove specified columns from the dataset.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with removed columns
        """
        drop_columns = [
            "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
            "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
            "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"
        ]
        return data.drop(drop_columns, axis=1)

    @staticmethod
    def clean_data(data):
        """
        Clean the dataset by removing missing values and duplicates.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print("Handling missing values and duplicates...")
        data = data.dropna(axis=0, how='any')
        data = data.drop_duplicates(subset=None, keep="first")
        return data

    @staticmethod
    def encode_categorical_variables(data):
        """
        Encode categorical variables using dummy encoding.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical variables
        """
        categorical_columns = [
            'http.request.method', 'http.referer', 'http.request.version',
            'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname',
            'mqtt.topic'
        ]
        
        print("Encoding categorical variables...")
        for column in categorical_columns:
            if column in data.columns:
                dummies = pd.get_dummies(data[column], prefix=column)
                data = pd.concat([data, dummies], axis=1)
                data = data.drop(column, axis=1)
        return data

    @staticmethod
    def save_preprocessing_report(initial_info, final_info, save_dir):
        """
        Save preprocessing report to a file.
        
        Args:
            initial_info (str): Initial dataset information
            final_info (str): Final dataset information
            save_dir (str): Directory to save the report
        """
        preprocessing_report = initial_info + "\n" + final_info
        save_output_to_file(preprocessing_report, 'preprocessing_report.txt', save_dir)

    @staticmethod
    def save_preprocessed_data(data, save_dir):
        """
        Save preprocessed dataset to CSV file.
        
        Args:
            data (pd.DataFrame): Preprocessed dataset
            save_dir (str): Directory to save the dataset
        """
        print("Saving preprocessed dataset...")
        data.to_csv(os.path.join(save_dir, 'preprocessed_DNN.csv'), index=False, encoding='utf-8')

    @staticmethod
    def preprocess_data(data_path, save_dir):
        """
        Main preprocessing pipeline for the Edge-IIoT dataset.
        
        Args:
            data_path (str): Path to the DNN-EdgeIIoT-dataset.csv file
            save_dir (str): Directory to save the preprocessed data and reports
        
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Load and process data
        data = DataPreprocessing.load_dataset(data_path)
        initial_info = DataPreprocessing.get_data_info(data)
        
        # Apply preprocessing steps
        data = DataPreprocessing.remove_unwanted_columns(data)
        data = DataPreprocessing.clean_data(data)
        data = shuffle(data, random_state=42)
        data = DataPreprocessing.encode_categorical_variables(data)
        
        # Save results
        final_info = DataPreprocessing.get_data_info(data)
        DataPreprocessing.save_preprocessing_report(initial_info, final_info, save_dir)
        DataPreprocessing.save_preprocessed_data(data, save_dir)
        
        return data
