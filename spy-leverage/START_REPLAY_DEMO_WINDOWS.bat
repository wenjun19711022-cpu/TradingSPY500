@echo off
chcp 65001 >nul
title SPY 杠杆抄底灯 - 历史回放
cd /d "%~dp0"
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo 没有找到 Python。也可以不装 Python：直接用浏览器打开 web\demo.html 看演示。
  start "" "%~dp0web\demo.html"
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo 第一次运行：正在安装需要的组件，大约 2 到 5 分钟……
  %PY% -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -e ".[futu]" || ".venv\Scripts\python.exe" -m pip install -e ".[futu]" -i https://pypi.tuna.tsinghua.edu.cn/simple || goto :fail
)
rem 回放 2023-08-17（日线抄底日），30 倍速。换日期：在后面加 --date 2024-03-12
".venv\Scripts\python.exe" -m spylev.live.app --source replay --date 2023-08-17 --speed 30 --open %*
pause
exit /b 0
:fail
echo 安装失败。请把这个窗口截图发给 Claude。
rmdir /s /q .venv 2>nul
pause
exit /b 1
