# This code takes the important PGN data and puts it in JSON file for data extraction 

from io import StringIO
import chess # type: ignore
import chess.pgn # type: ignore
import re
import json
import pathlib
import os

def extract_features_from_game(game):
    board = game.board()
    captures = 0
    checks = 0
    promotions = 0
    early_queen_development = False
    white_castled = False
    black_castled = False
    move_number = 0

    for move in game.mainline_moves():
        move_number += 1
        # Check castling
        if board.is_kingside_castling(move) or board.is_queenside_castling(move):
            if board.turn == chess.WHITE:
                white_castled = True
            else:
                black_castled = True

        # Check capture
        if board.is_capture(move):
            captures += 1

        # Check promotion
        if move.promotion:
            promotions += 1

        # Check for queen development in first 10 moves
        if move_number <= 20:
            if board.piece_at(move.from_square).piece_type == chess.QUEEN:
                early_queen_development = True

        board.push(move)
        if board.is_check():
            checks += 1

    headers = game.headers
    return {
        "ECO": headers.get("ECO", ""),
        "Opening": headers.get("Opening", ""),
        "Moves": move_number,
        "Captures": captures,
        "Checks": checks,
        "Promotions": promotions,
        "WhiteCastled": white_castled,
        "BlackCastled": black_castled,
        "EarlyQueenDev": early_queen_development,
    }

def split_pgn_file_to_games(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()


    raw_games = content.strip().split('\n\n\n')
    

    games = [game.strip() for game in raw_games if game.strip()]
    
    return games


def extract_clk_times_from_pgn(game):
    clk_times = []
    node = game
    while node.variations:
        node = node.variations[0]
        comment = node.comment
        match = re.search(r"\[%clk ([0-9:\.]+)\]", comment)
        if match:
            clk_times.append(match.group(1))
    
    return clk_times

def extract_eval_from_pgn(game):
    evals = []
    node = game
    while node.variations:
        node = node.variations[0]
        comment = node.comment
        match = re.search(r"\[%eval ([^\]]+)\]", comment)
        if match:
            evals.append(match.group(1))
    return evals

pgn_path = pathlib.Path().resolve()
output_path = str(pgn_path) + r"\DataSetOutput\output.json"
pgn_path1 = str(pgn_path) + r"\DataSet\lichess_db_broadcast_2025-05.pgn"
pgn_path2 = str(pgn_path) + r"\DataSet\lichess_db_broadcast_2025-06.pgn"
pgn_path3 = str(pgn_path) + r"\DataSet\lichess_db_broadcast_2025-06.pgn"
games_vector = split_pgn_file_to_games(pgn_path1)
games_vector.extend(split_pgn_file_to_games(pgn_path2))
games_vector.extend(split_pgn_file_to_games(pgn_path3))

print(f"Total games extracted: {len(games_vector)}")

results = []
for pgnString in games_vector:
    try:
        pgn = StringIO(pgnString)
        game = chess.pgn.read_game(pgn)
        if game is None:
            continue
        clkPerGame = extract_clk_times_from_pgn(game)
        evalPerGame = extract_eval_from_pgn(game)
        game_result = game.headers['Result']
        white_rating = game.headers.get("WhiteElo")
        black_rating = game.headers.get("BlackElo")
        if clkPerGame and game_result and white_rating and black_rating and evalPerGame:
            game_data = {
                "White": game.headers.get("WhiteTitle"),
                "Black": game.headers.get("BlackTitle"), 
                "WhiteElo": white_rating,
                "BlackElo": black_rating,
                "Result": game_result,
                "ClockTimes": clkPerGame,
                "Eval": evalPerGame

            }
            game_data.update(extract_features_from_game(game))
            results.append(game_data)
    except Exception as e:
        continue  
        





os.makedirs(os.path.dirname(output_path), exist_ok=True)


with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(results, out_file, indent=2)

print(len(results))





 


    