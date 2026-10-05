"""SQLite log: every closed bar, every signal, every options/flow snapshot -> daily review + future research."""
import json, os, sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars(tf TEXT, t TEXT, o REAL, h REAL, l REAL, c REAL, v REAL, PRIMARY KEY(tf, t));
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, tf TEXT, side TEXT, kind TEXT, bar_t TEXT,
  prob REAL, price REAL, ext REAL, atr REAL, ctx TEXT, pushed INTEGER, UNIQUE(tf, side, kind, bar_t));
CREATE TABLE IF NOT EXISTS options(t TEXT PRIMARY KEY, spot REAL, data TEXT);
CREATE TABLE IF NOT EXISTS flows(t TEXT PRIMARY KEY, data TEXT);
"""


class Store:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path); self.db.executescript(SCHEMA); self.db.commit()

    def bars(self, tf, rows):
        self.db.executemany("INSERT OR REPLACE INTO bars VALUES(?,?,?,?,?,?,?)", [(tf,) + tuple(r) for r in rows]); self.db.commit()

    def event(self, day, tf, side, kind, bar_t, prob, price, ext, atr, ctx, pushed):
        try:
            self.db.execute("INSERT INTO events(day,tf,side,kind,bar_t,prob,price,ext,atr,ctx,pushed) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (day, tf, side, kind, bar_t, prob, price, ext, atr, json.dumps(ctx, ensure_ascii=False), int(pushed)))
            self.db.commit(); return True
        except sqlite3.IntegrityError:
            return False                                  # already logged (restart-safe)

    def seen(self, tf, side, kind, bar_t):
        return self.db.execute("SELECT 1 FROM events WHERE tf=? AND side=? AND kind=? AND bar_t=?", (tf, side, kind, bar_t)).fetchone() is not None

    def options(self, t, spot, data):
        self.db.execute("INSERT OR REPLACE INTO options VALUES(?,?,?)", (t, spot, json.dumps(data, ensure_ascii=False))); self.db.commit()

    def flows(self, t, data):
        self.db.execute("INSERT OR REPLACE INTO flows VALUES(?,?)", (t, json.dumps(data, ensure_ascii=False))); self.db.commit()
