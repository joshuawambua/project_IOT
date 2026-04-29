@echo off
echo ========================================
echo XGBoost Predictor Setup for Windows
echo ========================================
echo.

echo Installing Python packages...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo Commands:
echo   python control.py start   - Start predictor
echo   python control.py stop    - Stop predictor
echo   python control.py restart - Restart predictor
echo   python control.py status  - Check status
echo.
pause