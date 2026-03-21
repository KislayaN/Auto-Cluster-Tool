import os
import pandas as pd

class Load_Data:
    def __init__(self):
        self.dataset = None
        self.path = None
        
    def load_data(self, path):
        if not isinstance(path, str):
            raise ValueError("Path must be a string.")
        
        if os.path.exists(path):
            self.path = path
            self.dataset = pd.read_csv(path)
            return self.dataset
        else:
            return FileNotFoundError(f"File is not found at {path}")