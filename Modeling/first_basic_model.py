# Basic model only cares about wins and losses, predictes through position data, time data, and elo data(buckets). Uses Random forest Classifier
# 90 %
# player white, looks at game data. num of moves, eval (+10) {who is winning by what amount, searching through variations}, just behaviours (goal) 



import json
import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # better model
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import numpy as np

def pop_dir(path): # this also gets rid of the backslash
    
    a = len(path) - 1
    print(a)
    while path[a] != '\\' :
        a-=1
    new_path = path[:a]
    return new_path

# Load JSON
path = str(pathlib.Path().resolve())
path = pop_dir(path) 
path += r"\Main\processed_outputs\basic_model.json"
json_path = path
with open(json_path, "r") as f:
    data = json.load(f)
print("Total rows in basic_model.json:", len(data))
# Create DataFrame
df = pd.DataFrame(data).dropna()

# Keep only wins and losses (Result 0 or 2)
df = df[df["Result"] != 1]
print("Result label counts:", Counter(df["Result"]))
# Re-map labels: 0 → 0 (loss), 2 → 1 (win), 
df["label"] = df["Result"].map({0: 0, 2: 1})

# Features and label
X = df.drop(columns=["Result", "label"], errors="ignore")
y = df["label"]

# Split te=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


import numpy as np

# --- Gini importances (from the trained forest) ---
feat_names = X_train.columns
gini = model.feature_importances_
order = np.argsort(gini)[::-1]

print("\nTop features by Gini importance:")
for i in order:
    print(f"{feat_names[i]:<25} {gini[i]:.4f}")

from sklearn.inspection import permutation_importance

perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_mean = perm.importances_mean
order = np.argsort(perm_mean)[::-1]

print("\nTop features by permutation importance (test set):")
for i in order:
    print(f"{feat_names[i]:<25} {perm_mean[i]:.4f}")

    