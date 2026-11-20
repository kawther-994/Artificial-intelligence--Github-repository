import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
# Get the folder where main.py is located
BASE_DIR = Path(__file__).resolve().parent

# Load the dataset from the same folder
data = pd.read_csv(BASE_DIR / "car.data", names=columns)

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


# PART 6 - COMPARISON OF BOTH ALGORITHMS

# Create a comparison table
comparison = X_test.copy()

comparison["Actual class"] = y_test
comparison["Algorithm 1"] = y_pred_1
comparison["Algorithm 2"] = y_pred_2

# Display the first 10 predictions
print("\nCOMPARISON OF FIRST 10 PREDICTIONS:")
print(comparison.head(10).to_string())

# Compare the predicted class distributions
print("\nALGORITHM 1 - CLASS DISTRIBUTION:")
print(y_pred_1.value_counts())

print("\nALGORITHM 2 - CLASS DISTRIBUTION:")
print(pd.Series(y_pred_2).value_counts())


# PART 7 - PERFORMANCE EVALUATION

# Evaluate Algorithm 1
print("\n========== ALGORITHM 1 EVALUATION ==========")

accuracy_1 = accuracy_score(y_test, y_pred_1)

print("Accuracy:", accuracy_1)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_1, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, y_pred_1,
    labels=["unacc", "acc", "good", "vgood"]
))


# Evaluate Algorithm 2
print("\n========== ALGORITHM 2 EVALUATION ==========")

accuracy_2 = accuracy_score(y_test, y_pred_2)

print("Accuracy:", accuracy_2)

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_2, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test, y_pred_2,
    labels=["unacc", "acc", "good", "vgood"]
))


# PART 8 - FINAL ACCURACY COMPARISON

print("\n========== FINAL ACCURACY COMPARISON ==========")

print("Algorithm 1 (Basic):",
      round(accuracy_1 * 100, 2), "%")

print("Algorithm 2 (Advanced):",
      round(accuracy_2 * 100, 2), "%")

print("\nMACRO F1-SCORE COMPARISON:")

from sklearn.metrics import f1_score

f1_1 = f1_score(y_test, y_pred_1, average="macro")
f1_2 = f1_score(y_test, y_pred_2, average="macro")

print("Algorithm 1:", round(f1_1, 3))
print("Algorithm 2:", round(f1_2, 3))


# GRAPH 1 - ACCURACY COMPARISON

algorithms = ["Algorithm 1 (Basic)", "Algorithm 2 (Advanced)"]
accuracies = [accuracy_1 * 100, accuracy_2 * 100]

plt.figure(figsize=(8, 5))

plt.bar(algorithms, accuracies)

plt.title("Accuracy Comparison of Both Algorithms")
plt.xlabel("AI Algorithms")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.2f}%",
             ha="center")

plt.tight_layout()

# Create the graphs folder if it doesn't exist
GRAPH_DIR = BASE_DIR / "graphs"
GRAPH_DIR.mkdir(exist_ok=True)

# Save the graph
plt.savefig(GRAPH_DIR / "accuracy_comparison.png")

plt.show()

# GRAPH 2 - PERFORMANCE METRICS COMPARISON

from sklearn.metrics import precision_recall_fscore_support
import numpy as np

# Calculate metrics for Algorithm 1
precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(
    y_test, y_pred_1,
    labels=["unacc", "acc", "good", "vgood"],
    zero_division=0
)

# Calculate metrics for Algorithm 2
precision_2, recall_2, f1_2, _ = precision_recall_fscore_support(
    y_test, y_pred_2,
    labels=["unacc", "acc", "good", "vgood"],
    zero_division=0
)

# Prepare graph
metrics = ["Precision", "Recall", "F1-score"]

values_1 = [
    np.mean(precision_1),
    np.mean(recall_1),
    np.mean(f1_1)
]

values_2 = [
    np.mean(precision_2),
    np.mean(recall_2),
    np.mean(f1_2)
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(x - width/2, values_1, width,
        label="Algorithm 1 - Basic")

plt.bar(x + width/2, values_2, width,
        label="Algorithm 2 - Advanced")

plt.title("Performance Metrics Comparison")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Macro Average Score")

plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()

plt.savefig(GRAPH_DIR / "performance_metrics.png")
plt.show()
