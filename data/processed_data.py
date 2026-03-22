import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

correct_root = r'C:\Users\kisla\OneDrive\Desktop\Auto Cluster Tool'

if correct_root not in sys.path:
    sys.path.insert(0, correct_root)
    
from data.load import Load_Data

load_dataset = Load_Data()
dataset_path = 'C:/Users/kisla/Downloads/archive/wine_dataset.csv'
load_dataset.load_data(path=dataset_path)
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
        dataset = self.dataset.copy()
        
        with open(self.path, 'r') as f:
            sample = f.read(2048) # Read small sample
            try:
                dialect = csv.Sniffer().sniff(sample) # Check if it has consistent delimiter
                has_header = csv.Sniffer().has_header(sample)
                print(f"Valid CSV with delimiter '{dialect.delimiter}'. Header present: {has_header}")
            except csv.Error: 
                print("Not a valid csv format.")
                return False
            
        # Getting the target column in the dataset
        
        categorical_cols = dataset.select_dtypes(exclude=['float']).columns 
        
        if len(categorical_cols) > 0:
            target_col = dataset[categorical_cols].nunique().idxmin()
            y = dataset[target_col]
        else: 
            target_col = dataset.nunique().idxmin()
            y = dataset[target_col]
        
        self.target = y.name
            
        # Flling missing values
        self.columns = dataset.columns.to_list()
        
        # Getting the Feature columns
        self.feature_columns = dataset.drop(columns=[target_col]).columns.tolist()
        
        # Building the dataframe including target
        self.dataframe = pd.DataFrame(dataset, columns=self.columns)
        
        # Building the X dataframe
        self.X_dataframe = pd.DataFrame(dataset.drop(columns=[self.target]), columns=self.feature_columns)
        self.X_dataset = self.X_dataframe.to_numpy()
        
        # Getting the numeric and Categoric columns 
        self.numeric_cols = self.X_dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.X_dataframe.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Pipeline for Imputation and Scaling
        numeric_processor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler())
            # ('scaler', StandardScaler())
        ])

        # Impute -> Encode kind of flow
        categorical_processor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_processor, self.numeric_cols),
                ('cat', categorical_processor, self.categorical_cols)
            ],
            verbose_feature_names_out=False
        )
        
        self.processed_dataset = self.preprocessor.fit_transform(dataset)

        self.processed_df = pd.DataFrame(
            self.processed_dataset, 
            columns=self.preprocessor.get_feature_names_out()
        )
    
        # Getting features that contains outliers
    
        outlier_info = {}
        
        for col in self.feature_columns:
            Q1 = self.X_dataframe[col].quantile(0.25)
            Q3 = self.X_dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.X_dataframe[col] = self.X_dataframe[col].clip(lower=lower_bound, upper=upper_bound)
            outliers = self.X_dataframe[(self.X_dataframe[col] < lower_bound) | (self.X_dataframe[col] > upper_bound)]

            if not outliers.empty:
                outlier_info[col] = len(outliers)
                
        print("Features with outliers:", list(outlier_info.keys()))
        
        print(f"Preprocessing complete. Final shape: {self.processed_df.shape}")
        
        return self.processed_df