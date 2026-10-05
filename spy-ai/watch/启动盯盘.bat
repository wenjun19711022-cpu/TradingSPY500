@echo off
chcp 65001 >nul
title SPY 盯盘机器人
set PYTHONPATH=C:\Users\94868\spybt\pylib
cd /d C:\Users\94868\spybt\watch
py -3 spy_watch.py
pause
