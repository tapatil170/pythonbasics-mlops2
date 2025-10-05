"""
task1.py
Task 1 - Python Basics: loads iris.csv (creates it if missing), prints shape and mean of numeric columns.
"""

import os
import pandas as pd
from sklearn.datasets import load_iris

CSV_NAME = "iris.csv"

def create_iris_csv(path=CSV_NAME):
    """Create iris.csv from sklearn if file doesn't exist."""
    if os.path.exists(path):
        return
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    df.to_csv(path, index=False)
    print(f"Created {path} from sklearn.datasets.")

def load_dataset(path=CSV_NAME):
    """Load CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    return df

def dataset_summary(df):
    """Print a short summary of the dataframe."""
    print("\n--- Dataset Summary ---")
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head().to_string(index=False))

def main():
    create_iris_csv()
    df = load_dataset()
    # Print number of rows & columns
    rows, cols = df.shape
    print(f"\nShape of dataset: {rows} rows, {cols} columns")
    # Show mean of numeric columns
    print("\nMean of numeric columns:")
    print(df.mean(numeric_only=True).to_string())
    # Optional: mean per species (useful learning)
    print("\nMean per species (numeric columns):")
    print(df.groupby('species').mean())
    # Show small summary
    dataset_summary(df)

if __name__ == "__main__":
    main()
