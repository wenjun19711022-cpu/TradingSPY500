@echo off
chcp 65001 >nul
rem Local launcher for spy-leverage without a new venv: uses this machine's packages (pandas 2.2.3 in spybt\pylib)
rem and a futu -> moomoo shim (local_shims\futu). Usage: run_local.bat replay 2026-10-02   |   run_local.bat live
set "PYTHONUTF8=1"
set "PYTHONPATH=C:\Users\94868\spybt\TradingSPY500\local_shims;C:\Users\94868\spybt\pylib;C:\Users\94868\spybt\TradingSPY500\spy-leverage"
cd /d C:\Users\94868\spybt\TradingSPY500\spy-leverage
if /i "%1"=="live" (
  py -3 -m spylev.live.app --source futu --web-port 8766 --leverage 20
) else (
  py -3 -m spylev.live.app --source replay --date %2 --speed 120 --web-port 8766 --leverage 20
)
