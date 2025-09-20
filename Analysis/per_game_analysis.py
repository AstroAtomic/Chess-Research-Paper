"""
    Per game analysis, I created this to help visualize the database from lichess (I was getting tired of looking at numbers)
"""

import matplotlib.pyplot as plt  # type: ignore
import json
import pathlib

def plot_game_timing(game, game_index=0, save_dir=None):
    white_pre = game.get('white_times_pre', [])
    black_pre = game.get('black_times_pre', [])
    result = game.get('result', 'unknown')
    white_elo = game.get('White_ELO', None)
    black_elo = game.get('Black_ELO', None)
    evals = game.get('Eval_Float', [])


    if len(white_pre) < 2 or len(black_pre) < 2 or len(evals) < 1:
        print(f"Skipping game {game_index}: insufficient data.")
        return
    if white_elo is None or black_elo is None:
        print(f"Skipping game {game_index}: missing ELO data.")
        return


    min_len = min(len(white_pre), len(black_pre), len(evals)) - 1
    white_deltas = [white_pre[i] - white_pre[i + 1] for i in range(min_len)]
    black_deltas = [black_pre[i] - black_pre[i + 1] for i in range(min_len)]
    eval_trimmed = evals[:min_len]
    moves = list(range(1, min_len + 1))


    elo_diff = white_elo - black_elo
    title_map = {"1-0": "White Wins", "0-1": "Black Wins", "1/2-1/2": "Draw"}
    result_text = title_map.get(result, "Unknown")
    title = (f"Game {game_index + 1} – {result_text}\n"
             f"White ELO: {white_elo} | Black ELO: {black_elo} | Diff: {elo_diff}")


    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title(title)
    ax1.set_xlabel("Move Number")
    ax1.set_ylabel("% Time Spent per Move")

    ax1.plot(moves, white_deltas, label='White Time %', marker='o', color='blue')
    ax1.plot(moves, black_deltas, label='Black Time %', marker='x', linestyle='--', color='orange')


    ax2 = ax1.twinx()
    ax2.set_ylabel("Eval (Centipawns)")
    ax2.plot(moves, eval_trimmed, label='Eval', color='green', linewidth=1.5, alpha=0.7)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.grid(True)
    plt.tight_layout()

    # Save or show
    if save_dir:
        fname = save_dir / f"game_{game_index + 1}_{result.replace('/', '-')}_elo{elo_diff}.png"
        plt.savefig(fname)
        print(f"Saved: {fname}")
    plt.show()
    plt.close()

def main():
    def pop_dir(path): # this also gets rid of the backslash
        
        a = len(path) - 1
        print(a)
        while path[a] != '\\':
            a-=1
        new_path = path[:a]
        return new_path

    # Load JSON
    path = str(pathlib.Path().resolve())
    path = pop_dir(path) 
    path += r"\processed_outputs\features.json"
    json_path = path
    save_plots = False  # Set True to save images
    save_dir = pathlib.Path("./game_plots")
    if save_plots:
        save_dir.mkdir(exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        games = json.load(f)

    for idx, game in enumerate(games):
        plot_game_timing(game, idx, save_dir if save_plots else None)

if __name__ == "__main__":
    main()
