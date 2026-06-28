@echo off
title Passport WiFi Studio
echo ==========================================
echo    Starting Passport Photo WiFi Studio
echo ==========================================
echo.
echo Checking your network IP...
for /f "tokens=4" %%a in ('route print ^| find " 0.0.0.0"') do set IP=%%a
echo.
echo ------------------------------------------
echo  TO ACCESS FROM YOUR PHONE:
echo  1. Connect phone to same WiFi as this PC
echo  2. Open browser and type: http://%IP%:8501
echo ------------------------------------------
echo.
streamlit run web_studio.py --server.address 0.0.0.0
pause
