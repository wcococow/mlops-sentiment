@echo off
set PYTHONPATH=D:\Kate Downloads\ML Learning\Interview\mlops-sentiment
python src\train.py --data data\imdb.csv --output model.joblib
