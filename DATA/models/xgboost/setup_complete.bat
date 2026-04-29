@echo off
echo ========================================
echo XGBOOST COMPLETE SETUP
echo ========================================
echo.

echo Step 1: Checking environment...
python check_environment.py
echo.

echo Step 2: Installing requirements...
pip install pandas numpy scikit-learn xgboost joblib firebase-admin psutil
echo.

echo Step 3: Training model...
python train_xgboost.py
echo.

echo Step 4: Testing model...
python test_model.py
echo.

echo Step 5: Starting predictor...
python control.py start
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Commands you can use:
echo   python control.py status   - Check if running
echo   python control.py stop     - Stop predictor
echo   python control.py restart  - Restart predictor
echo.
pause
