import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

correct_root = r'C:\Users\kisla\OneDrive\Desktop\Everything\Auto Cluster Tool'

if correct_root not in sys.path:
    sys.path.insert(0, correct_root)
    
from load import Load_Data

load_datast = Load_Data()
path = load_datast.path
load_datast.load_data(path=path)
dataset = load_datast.dataset

class Preprocess:
    def __init__(self, dataset, path):
        self.dataset = dataset
        self.path = path
        self.columns = None
        
    def preprocess(self):
        is_valid = False
        with open(self.path, 'r') as f:
            sample = f.read(2048) # Read small sample
            try:
                dialect = csv.Sniffer().sniff(sample) # Check if it has consistent delimiter
                has_header = csv.Sniffer().has_header(sample)
                print(f"Valid CSV with delimiter '{dialect.delimiter}'. Header present: {has_header}")
                is_valid = True
            except csv.Error: 
                print("Not a valid csv format.")
                return False
            
        # Getting the target column in the dataset
        categorical_cols = dataset.select_dtypes(exclude=['float']).columns # Get the categoric columns
        if len(categorical_cols) > 0:
            target_col = dataset[categorical_cols].nunique().idxmin()
            y = dataset[target_col]
        
        else: 
            target_col = dataset.nunique().idxmin()
            y = dataset[target_col]
            
        # Flling missing values
        self.columns = self.dataset.columns.to_list()
        dataframe = pd.DataFrame(dataset, columns=self.columns)
        
        numeric_processor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # 2. Define the categorical flow (Impute -> Encode)
        categorical_processor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_processor, self.numeric_cols),
                ('cat', categorical_processor, self.categorical_cols)
            ]
        )

        self.processed_data = self.preprocessor.fit_transform(self.dataset)

        self.processed_df = pd.DataFrame(
            self.processed_data, 
            columns=self.preprocessor.get_feature_names_out()
        )
        
        return dataframe