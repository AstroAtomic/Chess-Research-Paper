import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import json
import pathlib

limit = 30
winner_deltas_avg = [0 for _ in range(limit)]
looser_deltas_avg = [0 for _ in range(limit)]
total = 0

def get_oppening_features(data):
    global winner_deltas_avg, looser_deltas_avg, total, limit

    def is_valid_delta(x):
        return 0 <= x <= 115  

    for game in data:
        white_times = game['white_times_pre']
        black_times = game['black_times_pre']
        result = game['result']

        if len(white_times) < limit + 1 or len(black_times) < limit + 1:
            continue
        if result == "1/2-1/2":
            continue

        white_deltas = []
        black_deltas = []
        valid = True

        for i in range(limit):
            wd = white_times[i] - white_times[i + 1]
            bd = black_times[i] - black_times[i + 1]

            if not (is_valid_delta(wd) and is_valid_delta(bd)):
                valid = False
                break

            white_deltas.append(wd)
            black_deltas.append(bd)

        if not valid:
            continue  

        if result == "1-0":
            winner_deltas_avg = [w + d for w, d in zip(winner_deltas_avg, white_deltas)]
            looser_deltas_avg = [l + d for l, d in zip(looser_deltas_avg, black_deltas)]
        elif result == "0-1":
            winner_deltas_avg = [w + d for w, d in zip(winner_deltas_avg, black_deltas)]
            looser_deltas_avg = [l + d for l, d in zip(looser_deltas_avg, white_deltas)]

        total += 1

# Load data
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
path += r"\processed_outputs\features.json"
json_path = path
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Run analysis
get_oppening_features(data)

# Final averages
if total > 0:
    winner_deltas_avg = [d / total for d in winner_deltas_avg]
    looser_deltas_avg = [d / total for d in looser_deltas_avg]

# Plotting
x = np.arange(1, limit + 1)  # Move numbers 1 to limit
y1 = np.array(winner_deltas_avg)
y2 = np.array(looser_deltas_avg)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, marker='o', label="Winner")
plt.plot(x, y2, marker='x', linestyle='--', label="Loser")

plt.xlabel("Move Number")
plt.ylabel("Avg % Time Spent on Move")
plt.title("Average Time Spent per Move (First 30 Moves)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
