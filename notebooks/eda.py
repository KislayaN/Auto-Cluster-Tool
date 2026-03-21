import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from data.load import Load_Data
from data.processed_data import Preprocess

loader = Load_Data()
raw_data = loader.load_data(path='C:/Users/kisla/Downloads/archive/wine_dataset.csv')

processor = Preprocess(dataset=raw_data, path=loader.path)
processor.preprocess()
dataframe_cols = processor.columns

processor.preprocess()

dataframe = processor.dataset

for col in dataframe_cols:
    print(f"\nDataset Median for {col} is {dataframe[col].median()}")
    print(f"Dataset describe for {col} is \n{dataframe[col].describe()}")

# Feature Distribution Plot 

fig, axes = plt.subplots(2, 7, figsize=(10, 18))
axes = axes.flatten()

for i, feature in enumerate(processor.feature_columns):
    axes[i].hist(processor.X_dataframe[feature], bins=20, edgecolor='black')
    axes[i].set_title(f"Distribution of {feature}")

plt.delaxes(axes.flatten()[-1])
plt.tight_layout()
plt.show()