@echo off
chcp 65001 >nul
title SPY 全套：盯盘机器人 + 抄底灯
rem 1) moomoo OpenD (if installed and not running)  2) spy-ai watcher (alerts, swing mode, review, dashboard)
rem 3) spy-leverage monitor web page on http://127.0.0.1:8766 . Nothing here places orders.
tasklist /FI "IMAGENAME eq moomoo_OpenD.exe" | find /I "moomoo_OpenD.exe" >nul || (if exist "%APPDATA%\moomoo_OpenD\moomoo_OpenD.exe" start "" "%APPDATA%\moomoo_OpenD\moomoo_OpenD.exe")
timeout /t 20 /nobreak >nul
start "SPY 盯盘机器人（别关）" /min cmd /c "set PYTHONPATH=C:\Users\94868\spybt\pylib&& cd /d C:\Users\94868\spybt\watch&& py -3 spy_watch.py"
start "SPY 抄底灯（别关）" /min cmd /c "%~dp0run_local.bat live"
timeout /t 8 /nobreak >nul
start "" http://127.0.0.1:8766
