import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

correct_root = r'C:\Users\kisla\OneDrive\Desktop\Auto Cluster Tool'

if correct_root not in sys.path:
    sys.path.insert(0, correct_root)
    
from data.load import Load_Data

load_dataset = Load_Data()
load_dataset.path = 'C:/Users/kisla/Downloads/archive/wine_dataset.csv'
load_dataset.load_data(path=load_dataset.path)
dataset = load_dataset.dataset

class Preprocess:
    def __init__(self, dataset, path):
        self.dataset = dataset
        self.dataframe = None
        self.path = path
        self.feature_columns = None
        self.target = None
        self.X_dataframe = None
        self.X_dataset = None
        
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
        
        self.target = y.name
            
        # Flling missing values
        self.columns = self.dataset.columns.to_list()
        
        self.feature_columns = self.dataset.drop(columns=[target_col]).columns.tolist()
        
        dataframe = pd.DataFrame(dataset, columns=self.columns)
        
        self.numeric_cols = self.dataset.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.dataset.select_dtypes(include=['object', 'bool']).columns.tolist()
        
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
        
        self.X_dataframe = dataframe.drop(columns=[self.target])
        self.X_dataset = self.X_dataframe.to_numpy()
        
        return self.dataframe