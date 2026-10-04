@echo off
chcp 65001 >nul
title SPY 杠杆抄底灯 - 实盘监控
cd /d "%~dp0"

rem ---- 1. find Python (3.10+) ----
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo 没有找到 Python。浏览器会打开下载页：安装 Python 3.12，安装时一定勾选 "Add python.exe to PATH"，装完再双击本文件。
  start "" https://www.python.org/downloads/
  pause
  exit /b 1
)

rem ---- 2. first run: private environment + packages ----
if not exist ".venv\Scripts\python.exe" (
  echo 第一次运行：正在安装需要的组件，大约 2 到 5 分钟……
  %PY% -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -e ".[futu]" || ".venv\Scripts\python.exe" -m pip install -e ".[futu]" -i https://pypi.tuna.tsinghua.edu.cn/simple || goto :fail
)

rem ---- 3. run: needs moomoo OpenD open and logged in ----
echo 正在连接 moomoo OpenD……（这个黑色窗口不要关，关掉就停止监控）
".venv\Scripts\python.exe" -m spylev.live.app --source futu --open %*
pause
exit /b 0

:fail
echo 安装失败。请把这个窗口截图发给 Claude。
rmdir /s /q .venv 2>nul
pause
exit /b 1
