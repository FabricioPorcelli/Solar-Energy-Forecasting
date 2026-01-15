@echo off
REM Run preprocessing
python src\preprocessing.py

REM Run feature engineering
python src\features.py

REM Train the model
python src\train.py

REM Evaluate the model
python src\evaluate.py

pause
