@echo off
chcp 65001 >nul
title SPY 复盘看板
set PYTHONPATH=C:\Users\94868\spybt\pylib
cd /d C:\Users\94868\spybt\watch
py -3 dashboard.py
start "" "C:\Users\94868\spybt\watch\reports\看板.html"
