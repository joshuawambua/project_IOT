@echo off
cd /d C:\Users\Joshua\Desktop\finals\DATA\models\xgboost
echo Starting web server on port 8080...
echo Open http://localhost:8080/xgboost_dashboard.html in your browser
echo Press Ctrl+C to stop
python -m http.server 8080
pause
