# Cares about draws as well
# 84%
import json
import pandas as pd
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

def pop_dir(path): # this also gets rid of the backslash
    
    a = len(path) - 1
    print(a)
    while path[a] != '\\' :
        a-=1
    new_path = path[:a]
    return new_path

path = str(pathlib.Path().resolve())
path = pop_dir(path) 
path += r"\Main\processed_outputs\elo_model.json"
json_path = path
with open(json_path, "r") as f:
    data = json.load(f)

print("Total rows in elo_model.json:", len(data))


df = pd.DataFrame(data).dropna()


print("Original Result label counts:", Counter(df["Result"]))


df["label"] = df["Result"].map({0: 0, 1: 1, 2: 2})


X = df.drop(columns=["Result", "label"], errors="ignore")
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Loss", "Draw", "Win"]))
