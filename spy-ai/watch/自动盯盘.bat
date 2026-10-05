@echo off
chcp 65001 >nul
title SPY 盯盘机器人
set PYTHONPATH=C:\Users\94868\spybt\pylib
tasklist /FI "IMAGENAME eq moomoo_OpenD.exe" | find /I "moomoo_OpenD.exe" >nul || (if exist "%APPDATA%\moomoo_OpenD\moomoo_OpenD.exe" start "" "%APPDATA%\moomoo_OpenD\moomoo_OpenD.exe")
timeout /t 20 /nobreak >nul
cd /d C:\Users\94868\spybt\watch
start "SPY 盯盘机器人（别关，收盘后自动退出）" /min py -3 spy_watch.py
