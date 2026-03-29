import csv
import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class Preprocess:
    def __init__(self, dataset, path):
        self.dataset = dataset
        self.dataframe = None
        self.path = path
        self.feature_columns = None
        self.target = None
        self.X_dataframe = None
        self.X_dataset = None
        
    def _find_target(self, dataset):
        categorical_cols = dataset.select_dtypes(exclude=['float']).columns
        
        if len(categorical_cols) > 0:
            candidate = dataset[categorical_cols].nunique().idxmin()
        else:
            candidate = dataset.nunique().idxmin()
        
        if dataset[candidate].nunique() <= 10:
            return candidate
        else:
            return None
        
    def preprocess(self):
        dataset = self.dataset.copy()
        
        # CSV validation
        with open(self.path, 'r') as f:
            sample = f.read(2048)
            try:
                dialect = csv.Sniffer().sniff(sample)
                has_header = csv.Sniffer().has_header(sample)
                print(f"Valid CSV with delimiter '{dialect.delimiter}'. Header: {has_header}")
            except csv.Error:
                print("Not a valid CSV format.")
                return False
        
        # Target detection
        target_col = self._find_target(dataset)
        
        if target_col is not None:
            self.target = target_col
            self.feature_columns = dataset.drop(columns=[target_col]).columns.tolist()
            self.X_dataframe = dataset.drop(columns=[target_col]).copy()
        else:
            print("No target column detected — treating all columns as features")
            self.target = None
            self.feature_columns = dataset.columns.tolist()
            self.X_dataframe = dataset.copy()
        
        self.columns = dataset.columns.tolist()
        self.dataframe = dataset.copy()
        self.X_dataset = self.X_dataframe.to_numpy()
        
        # Numeric and categorical cols
        self.numeric_cols = self.X_dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.X_dataframe.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Outlier clipping
        for col in self.numeric_cols:
            Q1 = self.X_dataframe[col].quantile(0.25)
            Q3 = self.X_dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            self.X_dataframe[col] = self.X_dataframe[col].clip(
                lower=Q1 - 1.5 * IQR, 
                upper=Q3 + 1.5 * IQR
            )
        
        # Pipelines
        numeric_processor = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

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
        
        self.processed_dataset = self.preprocessor.fit_transform(self.X_dataframe)
        self.processed_df = pd.DataFrame(
            self.processed_dataset,
            columns=self.preprocessor.get_feature_names_out()
        )
        
        self.X_dataframe = self.processed_df
        self.feature_columns = self.processed_df.columns.tolist()
        
        print("Preprocessing done")
        return self.processed_df