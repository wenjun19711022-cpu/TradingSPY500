@echo off
chcp 65001 >nul
set PYTHONPATH=C:\Users\94868\spybt\pylib
cd /d C:\Users\94868\spybt\watch
py -3 replay_test.py 2026-04-09
start "" "C:\Users\94868\spybt\watch\reports\2026-04-09.html"
pause
