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
path += r"\Main\processed_outputs\basic_model.json"
json_path = path
with open(json_path, "r") as f:
    data = json.load(f)

print("Total rows in basic_model.json:", len(data))


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



import numpy as np
import pandas as pd

label_names = {0: "Loss", 1: "Draw", 2: "Win"}


y_pred_s = pd.Series(y_pred, index=X_test.index, name="pred")


def print_example(idx, kind="correct"):
    true_lbl = int(y_test.loc[idx])
    pred_lbl = int(y_pred_s.loc[idx])
    proba = model.predict_proba(X_test.loc[[idx]])[0] 
    proba_str = ", ".join(
        f"{label_names[i]}={proba[i]:.3f}" for i in range(len(proba))
    )
    print(f"\nExample ({kind}): index={idx}")
    print(f"  True: {label_names[true_lbl]} ({true_lbl})")
    print(f"  Pred: {label_names[pred_lbl]} ({pred_lbl})")
    print(f"  Probabilities: {proba_str}")

    ex = X_test.loc[idx]

    try:
        s = (ex - X.mean()) / (X.std(ddof=0).replace(0, np.nan))
        top_feats = s.abs().sort_values(ascending=False).index[:20]  
        print("  Top ~20 feature values:")
        print(ex[top_feats].to_string())
    except Exception:
        print("  Feature values:")
        print(ex.to_string())

# Indices for correct and incorrect sets
correct_idxs = [idx for idx in X_test.index if y_pred_s.loc[idx] == y_test.loc[idx]]
wrong_idxs   = [idx for idx in X_test.index if y_pred_s.loc[idx] != y_test.loc[idx]]

# Print one correct example (if any)
if len(correct_idxs) > 0:
    print_example(correct_idxs[0], kind="correct")
else:
    print("\nNo correctly predicted examples found on this test split (rare but possible).")

# Print one incorrect example (if any)
if len(wrong_idxs) > 0:
    print_example(wrong_idxs[0], kind="incorrect")
else:
    print("\nNo misclassified examples found on this test split 🎉.")    
