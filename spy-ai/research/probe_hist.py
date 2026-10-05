"""Read-only probe: how much SPY history (with real volume) moomoo OpenAPI serves."""
import moomoo as F
q = F.OpenQuoteContext(host="127.0.0.1", port=11111)
try:
    print("quota", q.get_history_kl_quota(get_detail=False))
    for ktype, start, end in ((F.KLType.K_1M, "2016-01-04", "2016-01-06"), (F.KLType.K_1M, "2019-01-02", "2019-01-03"),
                              (F.KLType.K_1M, "2021-01-04", "2021-01-05"), (F.KLType.K_1M, "2026-09-30", "2026-09-30"),
                              (F.KLType.K_5M, "2016-01-04", "2016-01-05")):
        ret, d, key = q.request_history_kline("US.SPY", start=start, end=end, ktype=ktype, max_count=1000, extended_time=False)
        if ret == F.RET_OK:
            print(ktype, start, len(d), d.time_key.iloc[0] if len(d) else None, d.time_key.iloc[-1] if len(d) else None,
                  d[["open", "high", "low", "close", "volume", "turnover"]].head(2).to_dict("records"))
        else:
            print(ktype, start, "ERR", d)
    print("quota after", q.get_history_kl_quota(get_detail=False))
finally:
    q.close()
