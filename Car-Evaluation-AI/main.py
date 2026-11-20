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
# DATA PREPARATION

# Separate input features (X) and target (y)
X = data.drop("class", axis=1)
y = data["class"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the dataset split
print("\nTRAINING AND TESTING DATA:")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


# ALGORITHM 1 - BASIC RULE-BASED AI

class BasicRuleBasedAI:

    def predict(self, row):

        # Rule 1: Low safety means unacceptable
        if row["safety"] == "low":
            return "unacc"

        # Rule 2: Cars with only 2 passenger seats
        elif row["persons"] == "2":
            return "unacc"

        # Rule 3: Very good car conditions
        elif (row["safety"] == "high"
              and row["persons"] == "more"
              and row["buying"] == "low"
              and row["maint"] == "low"):
            return "vgood"

        # Rule 4: Good safety and affordability
        elif (row["safety"] == "high"
              and row["persons"] in ["4", "more"]
              and row["buying"] in ["low", "med"]
              and row["maint"] in ["low", "med"]):
            return "good"

        # Rule 5: High safety and sufficient space
        elif (row["safety"] == "high"
              and row["persons"] in ["4", "more"]):
            return "acc"

        # Rule 6: Medium safety and affordable conditions
        elif (row["safety"] == "med"
              and row["persons"] in ["4", "more"]
              and row["buying"] == "low"
              and row["maint"] == "low"):
            return "acc"

        # Default rule
        else:
            return "unacc"


# Create Algorithm 1
model1 = BasicRuleBasedAI()

# Generate predictions using the test data
y_pred_1 = X_test.apply(model1.predict, axis=1)

# Display the first five predictions
print("\nALGORITHM 1 - BASIC RULE-BASED AI:")
print(y_pred_1.head())


# ALGORITHM 2 - ADVANCED RULE-BASED AI

# Convert categorical data into numerical values
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), X.columns)
    ]
)

# Create the advanced Decision Tree model
model2 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=6,
        class_weight="balanced",
        random_state=42
    ))
])

# Train the model using training data
model2.fit(X_train, y_train)

# Generate predictions using testing data
y_pred_2 = model2.predict(X_test)

# Display the first five predictions
print("\nALGORITHM 2 - ADVANCED RULE-BASED AI:")
print(y_pred_2[:5])

# DISPLAY THE DECISION TREE RULES

feature_names = model2.named_steps[
    "preprocessor"
].get_feature_names_out()

rules = export_text(
    model2.named_steps["classifier"],
    feature_names=list(feature_names)
)

print("\nADVANCED ALGORITHM - GENERATED RULES:")
print(rules)
