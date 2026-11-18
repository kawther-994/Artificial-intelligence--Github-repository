import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the dataset column names
columns = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
    "class"
]

# Load the dataset
data = pd.read_csv("car.data", names=columns)

# Display the first five records
print("FIRST FIVE RECORDS:")
print(data.head())
# DATASET EXPLORATION

# Display the dataset size
print("\nDATASET SIZE:")
print(data.shape)

# Display the distribution of car classes
print("\nCAR CLASS DISTRIBUTION:")
print(data["class"].value_counts())

# Check for missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# Display dataset information
print("\nDATASET INFORMATION:")
data.info()

# Display the column names
print("\nDATASET COLUMNS:")
print(data.columns.tolist())