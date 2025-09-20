import json
import pathlib
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore

def pop_dir(path): # this also gets rid of the backslash
    a = len(path) - 1
    print(a)
    while path[a] != '\\' :
        a-=1
    new_path = path[:a]
    return new_path
path = str(pathlib.Path().resolve())
path = pop_dir(path) 
path += r"\processed_outputs\basic_model.json"
json_path = path
with open(json_path, 'r', encoding='utf-8') as f:
    games = json.load(f)

df  = pd.DataFrame(games)

def plot_heatmap(games):
    numeric_df = games.drop(columns=['Result', 'label'], errors='ignore')
    corr = numeric_df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()



def _corr_heatmap(df_subset, title):
    # keep numeric cols (incl. booleans) but drop identifiers/labels you don’t want
    num = df_subset.select_dtypes(include=['number','bool']).drop(columns=['Result','label'], errors='ignore')
    if len(num) < 3 or num.shape[0] < 5:
        print(f"[{title}] Not enough games to compute meaningful correlations.")
        return
    corr = num.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_outcome_heatmaps(df):
    # Assumes Result: 0 = loss (for White), 1 = draw, 2 = win (for White)
    wins   = df[df['Result'] == 2]
    losses = df[df['Result'] == 0]
    draws  = df[df['Result'] == 1]

    _corr_heatmap(wins,   "Feature Correlations — White Wins")
    _corr_heatmap(losses, "Feature Correlations — White Losses")
    _corr_heatmap(draws,  "Feature Correlations — Draws")

plot_heatmap(df)
plot_outcome_heatmaps(df)

