#!/bin/bash
# SPY 杠杆抄底灯 - 实盘监控（macOS 双击运行；Linux 用 bash START_MAC.command）
# 加 --replay 改为历史回放：bash START_MAC.command --replay
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "没有找到 Python 3。请安装：https://www.python.org/downloads/ ，装完再双击本文件。"
  (open https://www.python.org/downloads/ 2>/dev/null || true)
  read -r -p "按回车退出" _; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "第一次运行：正在安装需要的组件，大约 2 到 5 分钟……"
  python3 -m venv .venv || { echo "创建环境失败"; read -r _; exit 1; }
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -e ".[futu]" \
    || .venv/bin/python -m pip install -e ".[futu]" -i https://pypi.tuna.tsinghua.edu.cn/simple \
    || { echo "安装失败，请把这个窗口截图发给 Claude。"; rm -rf .venv; read -r _; exit 1; }
fi
if [ "$1" = "--replay" ]; then
  shift
  exec .venv/bin/python -m spylev.live.app --source replay --date 2023-08-17 --speed 30 --open "$@"
fi
echo "正在连接 moomoo OpenD……（这个窗口不要关，关掉就停止监控）"
exec .venv/bin/python -m spylev.live.app --source futu --open "$@"
