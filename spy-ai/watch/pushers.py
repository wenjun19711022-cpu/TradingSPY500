"""Alerts: desktop (Windows toast + sound, default), PushPlus / Server酱 / 企业微信 (WeChat), or console only."""
import base64, json, subprocess, sys, urllib.parse, urllib.request

PS_AUMID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def _xml(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def desktop(title, text):
    """Windows 10/11 toast via PowerShell (no extra packages) + a short sound. Non-blocking."""
    text = text.replace("**", "")
    script = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$x = New-Object Windows.Data.Xml.Dom.XmlDocument
$x.LoadXml('<toast scenario="reminder"><visual><binding template="ToastGeneric"><text>%s</text><text>%s</text></binding></visual><audio src="ms-winsoundevent:Notification.Looping.Alarm2" loop="false"/><actions><action content="知道了" arguments="ok" activationType="foreground"/></actions></toast>')
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('%s').Show([Windows.UI.Notifications.ToastNotification]::new($x))
""" % (_xml(title).replace("'", "''"), _xml(text[:600]).replace("'", "''"), PS_AUMID)
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    try:
        subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return True
    except Exception as e:
        print("[桌面通知失败]", e); return False


def send(cfg, title, markdown):
    p = (cfg.get("push") or {}); prov = (p.get("provider") or "console").lower(); tok = (p.get("token") or "").strip()
    line = markdown.replace("\n\n", " | ")
    print("\a\n======== %s ========\n%s\n" % (title, line.replace(" | ", "\n")), flush=True)
    if prov == "desktop":
        return desktop(title, line.replace(" | ", "\n"))
    if prov == "console" or not tok or tok.startswith("在这里"):
        return False
    try:
        if prov == "pushplus":
            body = json.dumps({"token": tok, "title": title, "content": markdown, "template": "markdown"}).encode()
            req = urllib.request.Request("http://www.pushplus.plus/send", body, {"Content-Type": "application/json"})
        elif prov == "serverchan":
            req = urllib.request.Request("https://sctapi.ftqq.com/%s.send" % tok, urllib.parse.urlencode({"title": title, "desp": markdown}).encode())
        elif prov == "wecom":
            body = json.dumps({"msgtype": "markdown", "markdown": {"content": "**%s**\n%s" % (title, markdown)}}).encode()
            req = urllib.request.Request(tok, body, {"Content-Type": "application/json"})
        else:
            print("[推送] 未知 provider:", prov); return False
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read(); return True
    except Exception as e:
        print("[推送失败]", e, flush=True); return False


if __name__ == "__main__":
    ok = desktop("SPY 盯盘机器人 · 测试通知", "看到这条说明桌面提醒正常。\n真实信号示例：SPY 5分钟 底 70% | 收 674.75 低点 673.77 | 看跌墙 674（下方 1.1 ATR）")
    print("sent:", ok)
