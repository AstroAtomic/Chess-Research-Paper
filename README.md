# Chess Research Paper – Meta-Game Analysis

This repository contains the source code and scripts used for the research paper “Predicting Chess Outcomes from Meta-Game Data”. The project explores how contextual features such as rating differences, time usage, and activity statistics can be used to predict win/loss/draw outcomes using machine learning techniques.


## Setup

### Clone the repository:
```
git clone https://github.com/AstroAtomic/Chess-Research-Paper.git

cd Chess-Research-Paper
```
### Create folders:
```
mkdir Main 
cd Main
mkdir DataSet DataSetOutput processed_outputs
```
### Place these scripts in Main/:

tester.py

datasetExtractor.py

feature_vector.py

### Download three monthly PGN files from Lichess Broadcasts: 

https://database.lichess.org/#broadcasts

### To reproduce results from the paper, use:

2025-May

2025-June

2025-July

Place them in DataSet/.

## Running

Outputs are saved to DataSetOutput/ and processed_outputs/ for the first two scripts.

### Extract datasets:
python Main/datasetExtractor.py

### Generate feature vectors:
python Main/feature_vector.py

### Run experiments:
Run scripts in the other Folders

## Citation

If you use this code or dataset in your research, please cite:

Nitish Joson Terance Joe Heston. Chess Research Paper Repository. GitHub, 2025.
https://github.com/AstroAtomic/Chess-Research-Paper
