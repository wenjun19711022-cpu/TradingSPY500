"""Generate the moomoo main-chart formula 'SPY折线AI' (15m and 1h versions) + optional ZIG review line.
Core = the validated v5 bottom/top logic (bt4/moo_gen.py, identical to watch/v5core.py); added: pullback filter and
BUY/SELL alternation (see zv_formula.py, fitted to the user's drawing 2026-08-05..10-05)."""
import os, sys, json
sys.path.insert(0, "C:/Users/94868/spybt/bt4")
import moo_gen

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIRS = ["C:/Users/94868/OneDrive/桌面/交易策略/SPY_折线AI_moomoo", "C:/Users/94868/spybt/TradingSPY500/spy-ai/moomoo"]
DROP = ("STICKLINE", "DRAWTEXT", "LINE_B", "LINE_T", "ZERO", "TOP100", "BOT100", "SC:=")


def main_chart(N, D):
    core = [ln for ln in moo_gen.formula().splitlines() if ln and not ln.startswith(DROP)]
    add = ["N1:=%d;" % N, "D1:=%s;" % ("%g" % D),
           "HHN:=HHV(H,N1);", "LLN:=LLV(L,N1);",
           "BUY1:=FB AND (HHN-C)/HHN*100>=D1;",
           "SELL1:=FT AND (C-LLN)/LLN*100>=D1;",
           "BUY2:=BUY1 AND BARSLAST(SELL1)<BARSLAST(REF(BUY1,1));",
           "SELL2:=SELL1 AND BARSLAST(BUY1)<BARSLAST(REF(SELL1,1));",
           "DRAWTEXT(BUY2,L*0.998,'买'),COLORGREEN;",
           "DRAWTEXT(SELL2,H*1.002,'卖'),COLORRED;",
           "DRAWTEXT(FB AND BUY2=0,L*0.999,'·'),COLOR808080;",
           "DRAWTEXT(FT AND SELL2=0,H*1.001,'·'),COLOR808080;",
           "DRAWTEXT(OKB,L*0.996,'√'),COLORGREEN;",
           "DRAWTEXT(OKT,H*1.004,'√'),COLORRED;"]
    return "\n".join(core + add) + "\n"


ZIG = "ZZ:ZIG(3,0.45),COLORYELLOW,LINETHICK2;\n"

README = """SPY折线AI v1 — moomoo 主图指标（只提示，不下单）

文件
- SPY_折线AI_15分钟版.txt：贴到 15 分钟 K 线（推荐）。参数 N1=56、D1=0.3。
- SPY_折线AI_1小时版.txt：贴到 1 小时 K 线。参数 N1=7、D1=0。1 小时图上标记少，但更准。
- 可选_ZIG回看折线.txt：一行 ZIG(3,0.45)，画出和你手画的线差不多的折线（0.45% 转向）。
  这条线会"重画"：最后一段要等价格反向 0.45% 才定下来，只能用来复盘，不能当实时信号。
  moomoo 如果不支持 ZIG 函数，会提示不支持，删掉这一行就行。

图上的符号
- 绿色"买"：v5 底部判断成立，价格比近 N1 根 K 线的最高价回落了至少 D1%，而且上一个标记是"卖"。
- 红色"卖"：v5 顶部判断成立，价格比近 N1 根 K 线的最低价高出至少 D1%，而且上一个标记是"买"。
- 灰色"·"：v5 有底/顶判断，但没通过回落过滤或买卖交替（多半是你不会画的小波动）。
- "√"：事后确认（之后真的涨/跌了 2 个 ATR），比"买/卖"晚，只用来复盘。

和你手画的线有多吻合（2026-08-05 至 10-05，用 1 小时 0.45% 折线还原你的 39 个拐点）
- 15 分钟版：你画的底 58% 被标出，买点离你的底中位 0.18%；你画的顶 42% 被标出；约一半标记落在你画的点附近。
- 1 小时版：只标出 32% 的底，但几乎每个"买"都在你画的底上。
- v5/v6 的原始信号（不过滤）能覆盖你 89% 的底，代价是信号多一倍。

照着做能不能赚钱（OKX 手续费 12bp + 资金费，280 元保证金）
- 15 分钟版每个买点买、卖点卖：2024 年后每笔 -0.15%；50 倍每笔约 -42 元，358 笔里 65 笔爆仓。
- 1 小时版：2024 年后每笔 +0.14%；20 倍每笔约 +8 元；50 倍每笔约 -34 元，117 笔里 38 笔爆仓。
- 所以这个指标适合用来"看清楚现在在折线的哪一段"，不要每个标记都开 50 倍。
- 目前唯一回测站得住的做法是盯盘机器人的"波段模式"（抄底日 + 15 分钟/1 小时底进场，涨满 1 个 ATR 后见 15 分钟顶离场），上限 20 倍。

安装：moomoo 桌面端 → 指标 → 指标编辑 → 新建主图指标 → 粘贴 → 保存。
公式和电脑上的 Python 版本完全同一套计算（watch/v5core.py + v6/zv_formula.py），盯盘机器人的"折线买点/卖点"弹窗和图上的"买/卖"一致。
"""

if __name__ == "__main__":
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "SPY_折线AI_15分钟版.txt"), "w", encoding="utf-8").write(main_chart(56, 0.3))
        open(os.path.join(d, "SPY_折线AI_1小时版.txt"), "w", encoding="utf-8").write(main_chart(7, 0))
        open(os.path.join(d, "可选_ZIG回看折线.txt"), "w", encoding="utf-8").write(ZIG)
        open(os.path.join(d, "说明.md"), "w", encoding="utf-8").write(README)
        print("wrote", d)
    print(main_chart(56, 0.3)[-700:])
