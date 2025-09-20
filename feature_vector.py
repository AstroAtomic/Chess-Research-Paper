import json
import math
import pathlib
import numpy as np
from scipy.signal import find_peaks 
"""
Each feature vector will have these features:

(Keep in mind this holds ONLY time data, I will add the Rating barrier later if I can do it first with games)

-------Statistical analysis of various time indicators--------- (FOR WIN AND LOSS ONLY!)

White Elo:
Black Elo:

Numerics--- 

Avg Time Used by White TOTAL:
Avg Time Used by Black TOTAL:

Max time used by white TOTAL:
Max time used by black TOTAL:

Standard-Deviation by White TOTAL:
Standard-Deviation by Black TOTAL:



Precent---
conver times array to one with precent, first one being 1.0 for all 
Then you can easily calculate the same thing here!






Each Win/Draw/Loss vector will have these features:


Numerics--- 

Avg Time Used by Player TOTAL:
Max time used by Player TOTAL:
Standard-Deviation by Player TOTAL:
Entropy by Player TOTAL:
100-move data by Player TOTAL:



Precent---
conver times array to one with precent, first one being 1.0 for all 
Then you can easily calculate the same thing here!



"""

import math

def parse_eval(eval_str):
    if eval_str.startswith("#-"):
        return -50
    elif eval_str.startswith("#"):
        return 50
    else:
        try:
            return float(eval_str)
        except ValueError:
            return None


def clk_to_seconds(clk):
    parts = clk.split(":")
    #h is the hours, m is the mins, s is the seconds
    try:
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return 3600 * h + 60 * m + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return 60 * m + s
        elif len(parts) == 1:
            return int(parts[0]) if parts[0].isdigit() else 0
    except:
        return 0
    return 0

def convert_clkVec_to_secondsVec_White(clock_times):
    white_times = []
    i = 0
    while i < len(clock_times):
        white_times.append(clk_to_seconds(clock_times[i]))
        i += 2
    return white_times

def convert_STREVAL_to_NUM(Eval):
    newEval = []
    i = 0
    while i < len(Eval):
        newEval.append(parse_eval(Eval[i]))
        i+=1
    return newEval
    

def convert_clkVec_to_secondsVec_Black(clock_times):
    black_times = []
    i = 1
    while i < len(clock_times):
        black_times.append(clk_to_seconds(clock_times[i]))
        i += 2
    return black_times


def convert_num_to_Precent(array):
    pre_array = []
    first_element = array[0] if array else 1
    for element in array:
        if element != 0:
            pre_array.append((element / first_element) * 100)
        else:
            pre_array.append(0)
    return pre_array

def construct_delta(array):
    new_array = []
    prev = array[0]
    for element in array:
        new_array.append(prev - element)
        prev = element
    return new_array


def build_feature_vectors(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    cnt = 0

    features = []
    basic = []
    elo_mod = []
    for game in data:
        try:
            white = str(game['White'])
            black = str(game['Black'])
            if white == "BOT" or black == "BOT":
                cnt +=1
                continue
            result = str(game['Result'])
            white_won = 1 if result == "1-0" else -1 if result == "0-1" else 0 if result == "1/2-1/2" else 10
            if white_won == 10:
                cnt += 1
                continue

            white_elo = int(game['WhiteElo'])
            black_elo = int(game['BlackElo'])
            clock_times = list(game['ClockTimes'])
            Eval = list(game['Eval'])
            Moves = int(game['Moves'])
            Captures = int(game['Captures'])
            Checks = int(game['Checks'])
            Promotions = int(game['Promotions'])
            WhiteCastled = bool(game['WhiteCastled'])
            BlackCastled = bool(game['BlackCastled'])
            EarlyQueenDev = bool(game['EarlyQueenDev'])
            # Time vectors

            #numerics
            white_times_num = convert_clkVec_to_secondsVec_White(clock_times)
            black_times_num = convert_clkVec_to_secondsVec_Black(clock_times)
            Eval_Float = convert_STREVAL_to_NUM(Eval)


            white_times_pre = convert_num_to_Precent(white_times_num)
            black_times_pre = convert_num_to_Precent(black_times_num)

            num_moves = len(black_times_num)

            game_Data = {
                "White_ELO": white_elo,
                "Black_ELO": black_elo,
                "white_times_num": white_times_num,
                "black_times_num": black_times_num,
                "Eval_Float": Eval_Float,
                "white_times_pre": white_times_pre,
                "black_times_pre": black_times_pre,
                "result": result,
            }
            features.append(game_Data)
            white_delta_num = construct_delta(white_times_num)
            black_delta_num = construct_delta(black_times_num)
            if white_delta_num is None or black_delta_num is None:
                continue

            white_delta_num_avg = sum(white_delta_num) / len(white_delta_num)
            white_delta_num_std = np.std(white_delta_num)

            black_delta_num_avg = sum(black_delta_num) / len(black_delta_num)
            black_delta_num_std = np.std(black_delta_num)


            white_peaks, _ = find_peaks(white_delta_num, prominence=10)  
            black_peaks, _ = find_peaks(black_delta_num, prominence=10)
            elo_model = {
                "White_ELO": white_elo,
                "Black_ELO": black_elo,
                "Result": 0 if result == "0-1" else 1 if result == "1/2-1/2" else 2
            }
            basic_model = {
                "White_ELO": white_elo,
                "Black_ELO": black_elo,
                "white_delta_num_avg": white_delta_num_avg,
                "black_delta_num_avg": black_delta_num_avg,
                "white_delta_num_std": white_delta_num_std,
                "black_delta_num_std": black_delta_num_std,
                "Max_time_White": max(white_delta_num),
                "Max_time_Black": max(black_delta_num),
                "White_Peaks": len(white_peaks),
                "Black_Peaks": len(black_peaks),
                "Captures": Captures,
                "Moves": Moves,
                "Checks": Checks, 
                "Promotions": Promotions,
                "WhiteCastled": WhiteCastled,
                "BlackCastled": BlackCastled,
                "EarlyQueenDev": EarlyQueenDev,
                "Result": 0 if result == "0-1" else 1 if result == "1/2-1/2" else 2
            }
            basic.append(basic_model)


            elo_mod.append(elo_model) 



        

        except Exception as e:
            print(f"skipped due to {e}")
            cnt+=1

    print(f"{cnt} invalid results detected")
    return     features, basic, elo_mod

path = str(pathlib.Path().resolve()) 
features, basic, elo_mod = build_feature_vectors(path + r"\DataSetOutput\output.json")
output_path_features = path + r"\processed_outputs\features.json"
first_basic_model_data = path + r"\processed_outputs\basic_model.json"
elo_model_data = path + r"\processed_outputs\elo_model.json"

with open(output_path_features, "w", encoding="utf-8") as out_file:
    json.dump(features, out_file, indent=2)
with open(first_basic_model_data, "w", encoding="utf-8") as out_file:
    json.dump(basic, out_file, indent=2)
with open(elo_model_data, "w", encoding="utf-8") as out_file:
    json.dump(elo_mod, out_file, indent=2)


print(f"{len(features)} valid feature results")
print(f"{len(basic)} valid basic results")
print(f"{len(elo_mod)} valid elo_mod results")
      