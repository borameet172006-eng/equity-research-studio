import warnings, datetime as dt, math, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf

NA = "N/A"
META = {}

# ---------------------------------------------------------------- helpers
def isnum(x):
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x)) and np.isfinite(float(x))
    except Exception:
        return False

def num(x, d=2):
    return f"{float(x):,.{d}f}" if isnum(x) else NA

def pct(x, d=2):
    return f"{float(x):,.{d}f}%" if isnum(x) else NA

def safe_div(a, b):
    try:
        return float(a) / float(b) if isnum(a) and isnum(b) and float(b) != 0 else np.nan
    except Exception:
        return np.nan

def div_yield(info):
    v = info.get("trailingAnnualDividendYield")
    if isnum(v): return v * 100
    v = info.get("dividendYield")
    return v if isnum(v) else np.nan

def cagr(first, last, years):
    if isnum(first) and isnum(last) and first > 0 and last > 0 and years > 0:
        return ((last / first) ** (1 / years) - 1) * 100
    return np.nan

class Fmt:
    def __init__(self, currency, symbol_hint=""):
        self.cur = currency or "USD"
        self.inr = self.cur == "INR" or symbol_hint.endswith((".NS", ".BO"))
        self.sym = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}.get(self.cur, self.cur + " ")

    def price(self, v):
        return f"{self.sym}{v:,.2f}" if isnum(v) else NA

    def big(self, v):
        if not isnum(v): return NA
        v = float(v)
        if self.inr:
            return f"{self.sym}{v/1e7:,.2f} Cr" if abs(v) < 1e12 else f"{self.sym}{v/1e12:,.2f} L Cr"
        if abs(v) >= 1e12: return f"{self.sym}{v/1e12:,.2f}T"
        if abs(v) >= 1e9:  return f"{self.sym}{v/1e9:,.2f}B"
        return f"{self.sym}{v/1e6:,.2f}M"

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join([":---"] + ["---:"] * (len(headers) - 1)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

# ---------------------------------------------------------------- data
def _norm(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = pd.to_datetime(df.columns, errors="coerce")
    return df.reindex(sorted(df.columns, reverse=True), axis=1)

def _safe(fn, default=None):
    try:
        r = fn()
        return default if r is None else r
    except Exception:
        return default

def fetch_all(symbol):
    t = yf.Ticker(symbol)
    d = {"symbol": symbol}
    d["info"] = _safe(lambda: t.info, {}) or {}
    d["income"] = _norm(_safe(lambda: t.financials))
    d["balance"] = _norm(_safe(lambda: t.balance_sheet))
    d["cash"] = _norm(_safe(lambda: t.cashflow))
    d["q_income"] = _norm(_safe(lambda: t.quarterly_financials))
    d["q_cash"] = _norm(_safe(lambda: t.quarterly_cashflow))
    d["hist"] = _safe(lambda: t.history(period="5y", auto_adjust=False), pd.DataFrame())
    if d["hist"] is not None and not d["hist"].empty and d["hist"].index.tz is not None:
        d["hist"].index = d["hist"].index.tz_localize(None)
    d["dividends"] = _safe(lambda: t.dividends, pd.Series(dtype=float))
    d["inst"] = _safe(lambda: t.institutional_holders)
    d["news"] = _safe(lambda: t.news, [])
    return d

def row(df, *aliases):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    idx = {re.sub(r"\W", "", str(i)).lower(): i for i in df.index}
    for a in aliases:
        k = re.sub(r"\W", "", a).lower()
        if k in idx:
            s = df.loc[idx[k]]
            if isinstance(s, pd.DataFrame): s = s.iloc[0]
            return pd.to_numeric(s, errors="coerce")
    return pd.Series(dtype=float)

def at(s, i):
    try:
        return float(s.iloc[i]) if len(s) > i and isnum(s.iloc[i]) else np.nan
    except Exception:
        return np.nan

def ttm(qdf, *aliases):
    s = row(qdf, *aliases).dropna()
    return float(s.iloc[:4].sum()) if len(s) >= 4 else np.nan

# ---------------------------------------------------------------- technicals
def add_indicators(h):
    h = h.copy()
    c, hi, lo, v = h["Close"], h["High"], h["Low"], h["Volume"]
    for n in (20, 50, 200):
        h[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    delta = c.diff()
    up, dn = delta.clip(lower=0), -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / dn.ewm(alpha=1/14, adjust=False).mean()
    h["RSI"] = 100 - 100 / (1 + rs)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    h["MACD"] = ema12 - ema26
    h["MACD_SIG"] = h["MACD"].ewm(span=9, adjust=False).mean()
    h["MACD_HIST"] = h["MACD"] - h["MACD_SIG"]
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    h["BB_MID"], h["BB_UP"], h["BB_LO"] = ma, ma + 2 * sd, ma - 2 * sd
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    h["ATR"] = tr.ewm(alpha=1/14, adjust=False).mean()
    h["VOL_MA20"] = v.rolling(20).mean()
    return h

def technical_summary(h):
    last, prev = h.iloc[-1], h.iloc[-2]
    p = last["Close"]
    score = 0
    score += 1 if p > last["EMA200"] else -1
    score += 1 if last["EMA20"] > last["EMA50"] else -1
    score += 1 if last["MACD"] > last["MACD_SIG"] else -1
    if last["RSI"] > 70: score -= 1
    elif last["RSI"] < 30: score += 1
    stance = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"
    cross = "Bullish crossover (MACD > Signal)" if last["MACD"] > last["MACD_SIG"] else "Bearish (MACD < Signal)"
    if prev["MACD"] <= prev["MACD_SIG"] and last["MACD"] > last["MACD_SIG"]: cross = "Fresh bullish crossover"
    if prev["MACD"] >= prev["MACD_SIG"] and last["MACD"] < last["MACD_SIG"]: cross = "Fresh bearish crossover"
    rsi_sig = "Overbought (>70)" if last["RSI"] > 70 else "Oversold (<30)" if last["RSI"] < 30 else "Neutral"
    atr_pct = last["ATR"] / p * 100
    vol_ctx = "High volatility" if atr_pct > 3 else "Moderate volatility" if atr_pct > 1.5 else "Low volatility"
    bb = "Near upper band" if p >= last["BB_UP"] * 0.98 else "Near lower band" if p <= last["BB_LO"] * 1.02 else "Inside bands"
    vr = safe_div(last["Volume"], last["VOL_MA20"])
    px5 = safe_div(p, h["Close"].iloc[-6]) - 1 if len(h) > 6 else np.nan
    v5 = safe_div(h["Volume"].iloc[-5:].mean(), last["VOL_MA20"])
    if isnum(px5) and isnum(v5):
        if px5 > 0 and v5 > 1.1: vt = "Accumulation (price up on rising volume)"
        elif px5 < 0 and v5 > 1.1: vt = "Distribution (price down on rising volume)"
        elif px5 > 0: vt = "Weak rally (price up on fading volume)"
        else: vt = "Quiet pullback (price down on fading volume)"
    else:
        vt = NA
    return dict(stance=stance, score=score, cross=cross, rsi_sig=rsi_sig, atr_pct=atr_pct, vol_ctx=vol_ctx,
                bb=bb, vol_ratio=vr, vol_trend=vt, last=last, low52=h["Low"].iloc[-252:].min(),
                high52=h["High"].iloc[-252:].max())

# ---------------------------------------------------------------- fundamentals
def fundamentals(d):
    inc, bal, cf, qi, qc, info = d["income"], d["balance"], d["cash"], d["q_income"], d["q_cash"], d["info"]
    rev = row(inc, "Total Revenue", "Operating Revenue")
    ebit = row(inc, "EBIT", "Operating Income")
    da = row(inc, "Reconciled Depreciation", "Depreciation And Amortization")
    ebitda = row(inc, "EBITDA", "Normalized EBITDA")
    if ebitda.dropna().empty and not ebit.dropna().empty:
        ebitda = ebit.add(da.reindex(ebit.index).fillna(0), fill_value=0)
    ni = row(inc, "Net Income", "Net Income Common Stockholders")
    cogs = row(inc, "Cost Of Revenue")
    intexp = row(inc, "Interest Expense", "Interest Expense Non Operating")
    sga = row(inc, "Selling General And Administration")
    cfo = row(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")
    capex = row(cf, "Capital Expenditure")
    fcf = row(cf, "Free Cash Flow")
    if fcf.dropna().empty and not cfo.dropna().empty:
        fcf = cfo.add(capex.reindex(cfo.index).fillna(0), fill_value=0)

    ta = row(bal, "Total Assets")
    eq = row(bal, "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest")
    tl = row(bal, "Total Liabilities Net Minority Interest")
    debt = row(bal, "Total Debt")
    cl = row(bal, "Current Liabilities")
    ca = row(bal, "Current Assets")
    recv = row(bal, "Accounts Receivable", "Receivables")
    inv = row(bal, "Inventory")
    pay = row(bal, "Accounts Payable", "Payables")
    re_ = row(bal, "Retained Earnings")
    ppe = row(bal, "Net PPE")
    ltd = row(bal, "Long Term Debt")
    cash = row(bal, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    shares = row(bal, "Ordinary Shares Number", "Share Issued")

    ttm_rev, ttm_ebitda = ttm(qi, "Total Revenue"), ttm(qi, "EBITDA", "Normalized EBITDA")
    ttm_ni, ttm_cfo = ttm(qi, "Net Income", "Net Income Common Stockholders"), ttm(qc, "Operating Cash Flow")
    ttm_fcf = ttm(qc, "Free Cash Flow")

    def tri(s, ttm_v):
        return [at(s, 2), at(s, 1), ttm_v if isnum(ttm_v) else at(s, 0)]
    table = {"Revenue": tri(rev, ttm_rev), "EBITDA": tri(ebitda, ttm_ebitda), "PAT": tri(ni, ttm_ni),
             "CFO": tri(cfo, ttm_cfo), "FCF": tri(fcf, ttm_fcf)}

    r = {}
    r["roe"] = safe_div(at(ni, 0), at(eq, 0)) * 100
    r["roce"] = safe_div(at(ebit, 0), at(ta, 0) - at(cl, 0)) * 100
    r["de"] = safe_div(at(debt, 0), at(eq, 0))
    r["icr"] = safe_div(at(ebit, 0), abs(at(intexp, 0)) if isnum(at(intexp, 0)) else np.nan)
    r["fcf_cfo"] = safe_div(at(fcf, 0), at(cfo, 0))
    dso = safe_div(at(recv, 0), at(rev, 0)) * 365
    dio = safe_div(at(inv, 0), at(cogs, 0)) * 365
    dpo = safe_div(at(pay, 0), at(cogs, 0)) * 365
    r["ccc"] = (dso if isnum(dso) else 0) + (dio if isnum(dio) else 0) - (dpo if isnum(dpo) else 0) if (isnum(dso) or isnum(dio) or isnum(dpo)) else np.nan
    r["current"] = info.get("currentRatio") or safe_div(at(ca, 0), at(cl, 0))
    npm = safe_div(at(ni, 0), at(rev, 0))
    at_ = safe_div(at(rev, 0), at(ta, 0))
    lev = safe_div(at(ta, 0), at(eq, 0))
    r["dupont"] = dict(npm=npm * 100 if isnum(npm) else np.nan, at=at_, lev=lev,
                       roe=npm * at_ * lev * 100 if all(isnum(x) for x in (npm, at_, lev)) else np.nan)

    hist = d["hist"]
    h_pe, h_pb, h_ev = [], [], []
    if hist is not None and not hist.empty:
        for i, col in enumerate(rev.index if len(rev) else []):
            try:
                px = hist["Close"][:pd.Timestamp(col)].iloc[-1]
            except Exception:
                continue
            sh = at(shares, i) if isnum(at(shares, i)) else info.get("sharesOutstanding", np.nan)
            if not isnum(sh): continue
            eps_i, bv_i = safe_div(at(ni, i), sh), safe_div(at(eq, i), sh)
            if isnum(eps_i) and eps_i > 0: h_pe.append(px / eps_i)
            if isnum(bv_i) and bv_i > 0: h_pb.append(px / bv_i)
            e_val = px * sh + (at(debt, i) if isnum(at(debt, i)) else 0) - (at(cash, i) if isnum(at(cash, i)) else 0)
            if isnum(at(ebitda, i)) and at(ebitda, i) > 0: h_ev.append(e_val / at(ebitda, i))
    r["hist_pe"] = float(np.mean(h_pe)) if h_pe else np.nan
    r["hist_pb"] = float(np.mean(h_pb)) if h_pb else np.nan
    r["hist_ev"] = float(np.mean(h_ev)) if h_ev else np.nan

    mcap = info.get("marketCap", np.nan)
    z = np.nan
    try:
        wc = at(ca, 0) - at(cl, 0)
        z = (1.2 * wc / at(ta, 0) + 1.4 * at(re_, 0) / at(ta, 0) + 3.3 * at(ebit, 0) / at(ta, 0)
             + 0.6 * mcap / at(tl, 0) + 1.0 * at(rev, 0) / at(ta, 0))
        z = z if isnum(z) else np.nan
    except Exception:
        pass
    r["altman"] = z
    r["beneish"] = beneish(rev, cogs, recv, ca, ppe, ta, da, sga, cl, ltd, ni, cfo)

    raw = dict(rev=rev, ebitda=ebitda, ni=ni, cfo=cfo, fcf=fcf, capex=capex, debt=debt, cash=cash, shares=shares, eq=eq, ta=ta)
    return dict(table=table, ratios=r, raw=raw,
                fy={"Revenue": at(rev, 0), "EBITDA": at(ebitda, 0), "PAT": at(ni, 0)})

def beneish(rev, cogs, recv, ca, ppe, ta, da, sga, cl, ltd, ni, cfo):
    try:
        g = lambda s, i: at(s, i)
        dsri = (g(recv,0)/g(rev,0)) / (g(recv,1)/g(rev,1))
        gm = lambda i: (g(rev,i) - g(cogs,i)) / g(rev,i)
        gmi = gm(1) / gm(0)
        aq = lambda i: 1 - (g(ca,i) + g(ppe,i)) / g(ta,i)
        aqi = aq(0) / aq(1)
        sgi = g(rev,0) / g(rev,1)
        dep = lambda i: g(da,i) / (g(da,i) + g(ppe,i))
        depi = dep(1) / dep(0)
        sgai = (g(sga,0)/g(rev,0)) / (g(sga,1)/g(rev,1)) if isnum(g(sga,0)) and isnum(g(sga,1)) else 1.0
        lv = lambda i: (g(cl,i) + (g(ltd,i) if isnum(g(ltd,i)) else 0)) / g(ta,i)
        lvgi = lv(0) / lv(1)
        tata = (g(ni,0) - g(cfo,0)) / g(ta,0)
        m = (-4.84 + 0.920*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi + 0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi)
        return float(m) if isnum(m) else np.nan
    except Exception:
        return np.nan

# ---------------------------------------------------------------- DCF
def dcf_value(base_fcf, growth, wacc, tg, net_debt, shares, years=5):
    if not (isnum(base_fcf) and isnum(shares) and shares > 0) or wacc <= tg:
        return np.nan
    f, pv = base_fcf, 0.0
    for t in range(1, years + 1):
        g_t = growth + (tg - growth) * (t - 1) / max(years - 1, 1)
        f *= (1 + g_t)
        pv += f / (1 + wacc) ** t
    tv = f * (1 + tg) / (wacc - tg)
    pv += tv / (1 + wacc) ** years
    return (pv - (net_debt if isnum(net_debt) else 0)) / shares

def build_dcf(d, F, wacc, tg):
    info, raw = d["info"], F["raw"]
    base = ttm(d["q_cash"], "Free Cash Flow")
    if not isnum(base): base = at(raw["fcf"], 0)
    note = "FCF"
    if not isnum(base) or base <= 0:
        ni0 = at(raw["ni"], 0)
        base, note = (ni0 * 0.6 if isnum(ni0) and ni0 > 0 else np.nan), "60% of PAT (FCF negative/missing)"
    rv = raw["rev"].dropna()
    g_rev = cagr(rv.iloc[-1], rv.iloc[0], len(rv) - 1) / 100 if len(rv) >= 2 else np.nan
    growth = float(np.clip(g_rev if isnum(g_rev) else 0.08, 0.02, 0.15))
    shares = info.get("sharesOutstanding") or at(raw["shares"], 0)
    net_debt = (at(raw["debt"], 0) if isnum(at(raw["debt"], 0)) else 0) - (at(raw["cash"], 0) if isnum(at(raw["cash"], 0)) else 0)
    fv = dcf_value(base, growth, wacc, tg, net_debt, shares)
    waccs = [wacc - 0.02, wacc - 0.01, wacc, wacc + 0.01, wacc + 0.02]
    tgs = [max(tg + k, 0.0) for k in (-0.02, -0.01, 0, 0.01, 0.02)]
    grid = [[dcf_value(base, growth, w, g, net_debt, shares) for g in tgs] for w in waccs]
    return dict(fv=fv, growth=growth, base=base, note=note, waccs=waccs, tgs=tgs, grid=grid, net_debt=net_debt, shares=shares)

# ---------------------------------------------------------------- ownership, news, peers, sector
def ownership(d):
    info = d["info"]
    ins, inst = info.get("heldPercentInsiders"), info.get("heldPercentInstitutions")
    out = dict(promoter=ins * 100 if isnum(ins) else np.nan, inst=inst * 100 if isnum(inst) else np.nan,
               top_holders=[], shift="Quarter-over-quarter shift data is not provided by Yahoo; compare the NSE/BSE shareholding pattern (India) or 13F filings (US).")
    inst_df = d.get("inst")
    if isinstance(inst_df, pd.DataFrame) and not inst_df.empty:
        cols = {c.lower(): c for c in inst_df.columns}
        hc, pc, cc = cols.get("holder"), cols.get("pctheld") or cols.get("% out"), cols.get("pctchange")
        for _, r_ in inst_df.head(5).iterrows():
            out["top_holders"].append((r_.get(hc, NA), r_.get(pc, np.nan), r_.get(cc, np.nan)))
        if cc is not None:
            chg = pd.to_numeric(inst_df[cc], errors="coerce").dropna()
            if len(chg):
                out["shift"] = (f"Top reported institutions: {(chg > 0).sum()} increased vs {(chg < 0).sum()} reduced stakes "
                                f"in the latest filing (avg change {chg.mean()*100:+.2f}%).")
    return out

POS = {"beat", "beats", "surge", "surges", "growth", "profit", "record", "upgrade", "upgrades", "wins", "win", "order", "bullish", "rally",
       "gain", "gains", "strong", "outperform", "buy", "dividend", "expansion", "raises", "jump", "jumps", "soars", "approval"}
NEG = {"miss", "misses", "fall", "falls", "drop", "drops", "loss", "losses", "downgrade", "downgrades", "probe", "fraud", "penalty", "fine",
       "lawsuit", "bearish", "weak", "decline", "cut", "cuts", "slump", "plunge", "resigns", "default", "sell", "warning", "ban", "recall"}

def news_sentiment(d):
    heads = []
    for n in d.get("news") or []:
        t = (n.get("content", {}) or {}).get("title") or n.get("title")
        if t: heads.append(t)
    scored = []
    for h_ in heads[:10]:
        w = set(re.findall(r"[a-z]+", h_.lower()))
        scored.append((h_, len(w & POS) - len(w & NEG)))
    tot = sum(s for _, s in scored)
    label = "Bullish" if tot >= 2 else "Bearish" if tot <= -2 else "Neutral"
    return label, scored

PEER_MAP = {
    "Information Technology Services": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Banks - Regional": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
    "Oil & Gas Refining & Marketing": ["RELIANCE.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS"],
    "Auto Manufacturers": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "Drug Manufacturers - Specialty & Generic": ["SUNPHARMA.NS", "CIPLA.NS", "DRREDDY.NS", "LUPIN.NS"],
    "Consumer Electronics": ["AAPL", "SONY", "SSNLF"],
    "Software - Infrastructure": ["MSFT", "ORCL", "ADBE", "CRM"],
}

def peer_table(symbol, info, peers, fm):
    if not peers:
        peers = [p for p in PEER_MAP.get(info.get("industry", ""), []) if p.upper() != symbol.upper()][:4]
    rows = []
    for s in [symbol] + list(peers):
        try:
            i = info if s == symbol else (yf.Ticker(s).info or {})
            rows.append(dict(sym=s, pe=i.get("trailingPE"), pb=i.get("priceToBook"),
                             roe=(i.get("returnOnEquity") or np.nan) * 100, g=(i.get("revenueGrowth") or np.nan) * 100,
                             mc=i.get("marketCap")))
        except Exception:
            rows.append(dict(sym=s, pe=np.nan, pb=np.nan, roe=np.nan, g=np.nan, mc=np.nan))
    df = pd.DataFrame(rows)
    def rk(col, asc):
        return pd.to_numeric(df[col], errors="coerce").rank(ascending=asc, pct=True)
    df["score"] = (rk("pe", False) + rk("roe", True) + rk("g", True)).div(3).mul(100)
    return df

def sector_kpis(info, F, fm):
    sec = info.get("sector", "")
    emp = info.get("fullTimeEmployees")
    rev = at(F["raw"]["rev"], 0)
    om = pct((info.get("operatingMargins") or np.nan) * 100)
    k = []
    if sec == "Financial Services":
        k += [("Return on Assets", pct((info.get("returnOnAssets") or np.nan) * 100)), ("Price / Book", num(info.get("priceToBook"))),
              ("Dividend Yield", pct(div_yield(info))),
              ("NIM / Gross NPA / CASA", "Not in Yahoo data - read from the company's quarterly investor presentation")]
    elif sec == "Technology":
        k += [("Revenue / Employee", fm.big(safe_div(rev, emp)) if isnum(emp) else NA), ("Employees", f"{emp:,}" if emp else NA),
              ("Operating Margin", om), ("Attrition / TCV deals", "Not in Yahoo data - see quarterly results deck")]
    elif sec == "Healthcare":
        k += [("R&D intensity", NA), ("Operating Margin", om),
              ("USFDA approvals / ANDAs", "Not in Yahoo data - check USFDA database & company disclosures")]
    elif sec in ("Energy", "Utilities"):
        k += [("EV/EBITDA", num(info.get("enterpriseToEbitda"))), ("Dividend Yield", pct(div_yield(info)))]
    else:
        k += [("Gross Margin", pct((info.get("grossMargins") or np.nan) * 100)), ("Operating Margin", om),
              ("Revenue / Employee", fm.big(safe_div(rev, emp)) if isnum(emp) else NA)]
    return k

# ---------------------------------------------------------------- report
def status_vs(cur, hist, tol=0.10):
    if not (isnum(cur) and isnum(hist)): return NA
    d_ = cur / hist - 1
    return f"Premium ({d_*100:+.0f}%)" if d_ > tol else f"Discount ({d_*100:+.0f}%)" if d_ < -tol else "In line"

def generate_report(symbol, wacc=0.10, tg=0.04, peers=None, show_charts=True):
    symbol = symbol.strip().upper()
    d = fetch_all(symbol)
    info, hist = d["info"], d["hist"]
    if hist is None or hist.empty:
        return f"❌ No price data found for `{symbol}`. Use Yahoo tickers, e.g. `TCS.NS`, `RELIANCE.NS`, `AAPL`.", None
    fm = Fmt(info.get("currency") or info.get("financialCurrency"), symbol)
    H = add_indicators(hist.dropna(subset=["Close"]))
    T = technical_summary(H)
    F = fundamentals(d)
    R = F["ratios"]
    D = build_dcf(d, F, wacc, tg)
    if info.get("sector") == "Financial Services":
        D["fv"] = np.nan
        D["note"] = "DCF not meaningful for banks/NBFCs - use P/B vs ROE"
    O = ownership(d)
    sent, scored = news_sentiment(d)

    price = float(H["Close"].iloc[-1])
    pe = info.get("trailingPE"); pb = info.get("priceToBook"); evx = info.get("enterpriseToEbitda")
    eps = info.get("trailingEps")
    upside = (D["fv"] / price - 1) * 100 if isnum(D["fv"]) else np.nan

    rel_pe_val = R["hist_pe"] * eps if isnum(R["hist_pe"]) and isnum(eps) and eps > 0 else np.nan
    cands = [x for x in (D["fv"], rel_pe_val) if isnum(x)]
    tgt_lo, tgt_hi = (min(cands), max(cands)) if cands else (np.nan, np.nan)
    tgt_mid = float(np.mean(cands)) if cands else np.nan

    bits = []
    if isnum(pe) and isnum(R["hist_pe"]):
        rel = "premium" if pe > R["hist_pe"] * 1.1 else "discount" if pe < R["hist_pe"] * 0.9 else "in line"
        bits.append(f"trades at {pe:.1f}x earnings vs ~{R['hist_pe']:.1f}x own history ({rel})")
    elif isnum(pe): bits.append(f"trades at {pe:.1f}x trailing earnings")
    if isnum(upside): bits.append(f"DCF (WACC {wacc*100:.1f}%, g {tg*100:.1f}%) implies {upside:+.1f}% {'upside' if upside>0 else 'downside'}")
    bits.append(f"RSI {T['last']['RSI']:.0f} is {T['rsi_sig'].split(' ')[0].lower()}")
    verdict = ("accumulate on dips" if isnum(upside) and upside > 15 and T["stance"] != "Bearish" else
               "valuation looks stretched" if isnum(upside) and upside < -15 else "fairly valued / wait for confirmation")
    thesis = f"{info.get('longName', symbol)} {', '.join(bits)}; overall: {verdict}."

    L = []
    A = L.append
    A(f"# 📊 EQUITY RESEARCH REPORT: {symbol}")
    A(f"**Company Name:** {info.get('longName', NA)} | **Sector:** {info.get('sector', NA)} | **Industry:** {info.get('industry', NA)}  ")
    A(f"**Current Price:** {fm.price(price)} | **Market Cap:** {fm.big(info.get('marketCap'))} | **Date:** {dt.date.today():%d %b %Y}\n\n---\n")
    A("## EXECUTIVE SUMMARY")
    A(f"- **Investment Thesis:** {thesis}")
    A(f"- **Key Valuation Metrics:** P/E: **{num(pe)}** | EV/EBITDA: **{num(evx)}** | DCF Fair Value: **{fm.price(D['fv'])}** | Upside/Downside: **{pct(upside, 1)}**")
    A(f"- **Technical Stance:** **{T['stance']}** (200-EMA: {fm.price(T['last']['EMA200'])}, RSI: {T['last']['RSI']:.1f})")
    A(f"- **News Sentiment (headline lexicon, last {len(scored)}):** **{sent}**\n\n---\n")

    A("## DOMAIN 1: FUNDAMENTAL ANALYSIS & DUPONT BREAKDOWN\n")
    A("### 1.1 Financial Statements Summary")
    rows = []
    names = {"Revenue": "Revenue", "EBITDA": "Operating Profit (EBITDA)", "PAT": "Net Profit (PAT)", "CFO": "Cash Flow from Ops (CFO)", "FCF": "Free Cash Flow (FCF)"}
    for k, (a, b, c) in F["table"].items():
        cg = cagr(a, F["fy"].get(k, np.nan), 2) if k in ("Revenue", "EBITDA", "PAT") else np.nan
        rows.append([f"**{names[k]}**", fm.big(a), fm.big(b), fm.big(c), pct(cg, 1) if isnum(cg) else "-"])
    A(md_table(["Metric", "Year - 2", "Year - 1", "TTM / Latest", "CAGR (%)"], rows))
    A("\n*Year-2/Year-1 are fiscal years; the third column is TTM from quarterly data when available, else latest FY. CAGR is FY-2 to latest FY.*\n")

    dp = R["dupont"]
    A("### 1.2 DuPont Analysis (ROE Decomposition)\n```")
    A("ROE  =  Net Profit Margin  ×  Asset Turnover  ×  Financial Leverage")
    A(f"{num(dp['roe'])}%  =  {num(dp['npm'])}%  ×  {num(dp['at'])}x  ×  {num(dp['lev'])}x\n```")
    A(f"- **Profit Margin:** {pct(dp['npm'])} (PAT / Revenue)\n- **Asset Turnover:** {num(dp['at'])}x (Revenue / Total Assets)\n- **Financial Leverage:** {num(dp['lev'])}x (Total Assets / Equity)")
    if all(isnum(x) for x in (dp["npm"], dp["at"], dp["lev"])):
        if dp["lev"] > 4: dr = "leverage (high balance-sheet gearing) is a major ROE driver - quality of ROE is weaker"
        elif dp["npm"] > 15: dr = "high profit margins (pricing power / operating efficiency) are the main ROE driver"
        elif dp["at"] > 1: dr = "asset turnover (efficient capital use) is the main ROE driver"
        else: dr = "ROE is balanced across margin, turnover and leverage"
        A(f"- **DuPont Conclusion:** {dr}.\n")
    else:
        A(f"- **DuPont Conclusion:** {NA}\n")

    A("### 1.3 Health & Valuation Ratios")
    de_s = NA if not isnum(R["de"]) else ("Safe (<1.0)" if R["de"] < 1 else "Elevated")
    ic_s = NA if not isnum(R["icr"]) else ("Safe (>3.0)" if R["icr"] > 3 else "Weak")
    cr_s = NA if not isnum(R["current"]) else ("Comfortable" if R["current"] >= 1.5 else "Adequate" if R["current"] >= 1 else "Tight")
    A(md_table(["Ratio", "Current Value", "Hist. Avg (FY-end, approx.)", "Status / Benchmark"], [
        ["**P/E Ratio**", num(pe), num(R["hist_pe"]), status_vs(pe, R["hist_pe"])],
        ["**Price / Book (P/B)**", num(pb), num(R["hist_pb"]), status_vs(pb, R["hist_pb"])],
        ["**EV / EBITDA**", num(evx), num(R["hist_ev"]), status_vs(evx, R["hist_ev"])],
        ["**ROE**", pct(R["roe"]), "-", NA if not isnum(R["roe"]) else ("Strong (>15%)" if R["roe"] > 15 else "Moderate")],
        ["**ROCE**", pct(R["roce"]), "-", NA if not isnum(R["roce"]) else ("Strong (>15%)" if R["roce"] > 15 else "Moderate")],
        ["**Debt / Equity**", num(R["de"]), "-", de_s],
        ["**Interest Coverage**", num(R["icr"]) + ("x" if isnum(R["icr"]) else ""), "-", ic_s],
        ["**Current Ratio**", num(R["current"]), "-", cr_s],
        ["**Cash Conversion Cycle**", num(R["ccc"], 0) + (" days" if isnum(R["ccc"]) else ""), "-", "Lower is better"],
        ["**FCF / CFO**", num(R["fcf_cfo"]), "-", NA if not isnum(R["fcf_cfo"]) else ("Capital-light" if R["fcf_cfo"] > 0.7 else "Capex heavy")],
    ]))
    A("\n*Historical averages use the (up to 4) fiscal years Yahoo exposes with FY-end prices - an approximation of a 5-yr average.*\n")

    A("### 1.4 Earnings Quality & Distress Screens")
    z, m = R["altman"], R["beneish"]
    fin = info.get("sector") == "Financial Services"
    z_s = "Not applicable to banks/NBFCs" if fin else NA if not isnum(z) else ("Safe (>2.99)" if z > 2.99 else "Grey zone (1.81-2.99)" if z > 1.81 else "Distress (<1.81)")
    m_s = "Not applicable to banks/NBFCs" if fin else NA if not isnum(m) else ("Possible manipulation flag (> -1.78)" if m > -1.78 else "Unlikely manipulator (< -1.78)")
    A(md_table(["Model", "Score", "Interpretation"], [["**Altman Z-Score**", NA if fin else num(z), z_s], ["**Beneish M-Score**", NA if fin else num(m), m_s]]))

    A("\n### 1.5 DCF & Sensitivity (price per share)")
    A(f"Base cash flow: {fm.big(D['base'])} ({D['note']}), 5-yr growth fading from {D['growth']*100:.1f}% to terminal {tg*100:.1f}%, WACC {wacc*100:.1f}%.\n")
    hdr = ["WACC \\ Terminal g"] + [f"{g*100:.1f}%" for g in D["tgs"]]
    A(md_table(hdr, [[f"**{w*100:.1f}%**"] + [fm.price(v) for v in D["grid"][i]] for i, w in enumerate(D["waccs"])]))

    A("\n### 1.6 Peer Comparison & Relative Score")
    try:
        P = peer_table(symbol, info, peers, fm)
        if len(P) > 1:
            A(md_table(["Ticker", "P/E", "P/B", "ROE %", "Rev Growth %", "Mkt Cap", "Score /100"],
                       [[r_.sym, num(r_.pe), num(r_.pb), num(r_.roe), num(r_.g), fm.big(r_.mc), num(r_.score, 0)] for r_ in P.itertuples()]))
            A("\n*Score = average percentile rank of (lower P/E, higher ROE, higher revenue growth) within the group.*")
        else:
            A("No peers supplied/auto-detected - enter comma-separated peer tickers in the Peers box.")
    except Exception as e:
        A(f"Peer table unavailable ({e}).")

    A("\n---\n")
    l = T["last"]
    sig = lambda v: "Price above (support)" if price > v else "Price below (resistance)"
    A("## DOMAIN 2: TECHNICAL & VOLUME STRUCTURE\n\n### 2.1 Trend & Momentum Indicators")
    A(md_table(["Technical Indicator", "Value", "Signal"], [
        ["**20-Day EMA**", fm.price(l["EMA20"]), sig(l["EMA20"])],
        ["**50-Day EMA**", fm.price(l["EMA50"]), sig(l["EMA50"])],
        ["**200-Day EMA**", fm.price(l["EMA200"]), "Long-term uptrend" if price > l["EMA200"] else "Long-term downtrend"],
        ["**RSI (14)**", num(l["RSI"]), T["rsi_sig"]],
        ["**MACD / Signal**", f"{l['MACD']:.2f} / {l['MACD_SIG']:.2f}", T["cross"]],
        ["**Bollinger (20,2)**", f"{fm.price(l['BB_LO'])} – {fm.price(l['BB_UP'])}", T["bb"]],
        ["**ATR (14)**", f"{fm.price(l['ATR'])} ({T['atr_pct']:.2f}% of price)", T["vol_ctx"]],
        ["**52-Week Range**", f"{fm.price(T['low52'])} – {fm.price(T['high52'])}", f"{(price/T['high52']-1)*100:.1f}% from high"],
    ]))
    A("\n### 2.2 Volume & Delivery Assessment")
    A(f"- **Latest Volume vs 20-DMA Volume:** {num(T['vol_ratio'])}x ({'Spike' if isnum(T['vol_ratio']) and T['vol_ratio'] > 1.5 else 'Normal'})")
    A(f"- **Volume Trend:** {T['vol_trend']}")
    A("- *Delivery % is not available from Yahoo; use NSE bhavcopy for delivery-based confirmation.*\n\n---\n")

    A("## DOMAIN 3: INSTITUTIONAL FLOWS & CATALYSTS")
    A(f"- **Promoter / Insider Holding:** {pct(O['promoter'])}")
    A(f"- **Institutional Holding (FII + DII):** {pct(O['inst'])}")
    A(f"- **Insider / Major Shareholder Shift:** {O['shift']}")
    if O["top_holders"]:
        A("\n" + md_table(["Top Institutional Holder", "% Held", "Change"], [[h_, num(p_ * 100 if isnum(p_) else p_), pct(c_ * 100) if isnum(c_) else NA] for h_, p_, c_ in O["top_holders"]]))
    dv = d["dividends"]
    if dv is not None and len(dv):
        A(f"\n- **Dividends:** last {fm.price(float(dv.iloc[-1]))} on {dv.index[-1]:%d %b %Y}; trailing-12M total {fm.price(float(dv[dv.index > dv.index[-1] - pd.Timedelta(days=365)].sum()))}")
    if scored:
        A("\n**Recent headlines (lexicon score):**")
        for h_, s_ in scored[:5]:
            A(f"- {'🟢' if s_ > 0 else '🔴' if s_ < 0 else '⚪'} {h_}")
    A("\n---\n")

    inval = l["EMA200"] if price > l["EMA200"] else T["low52"]
    bull = "Revenue growth & margin expansion" if isnum(F["table"]["Revenue"][2]) else "Earnings recovery"
    rc = cagr(F["table"]["Revenue"][0], F["fy"]["Revenue"], 2)
    if isnum(rc) and rc > 10:
        bull = f"Sustained revenue growth (~{rc:.0f}% CAGR) with operating leverage"
    risks = []
    if isnum(R["de"]) and R["de"] > 1: risks.append("High debt")
    if isnum(upside) and upside < -10: risks.append("Valuation stretch vs DCF")
    if isnum(pe) and isnum(R["hist_pe"]) and pe > R["hist_pe"] * 1.15: risks.append("Premium to own historical P/E")
    if l["RSI"] > 70: risks.append("RSI overbought")
    if isnum(m) and m > -1.78 and not fin: risks.append("Earnings-quality flag (Beneish)")
    if isnum(z) and z < 1.81 and not fin: risks.append("Distress zone (Altman)")
    A("## 🎯 ACTIONABLE CONCLUSION & RISK MATRIX")
    A(f"- **Target Price Range (DCF + Relative):** **{fm.price(tgt_lo)} – {fm.price(tgt_hi)}** (mid {fm.price(tgt_mid)})")
    A(f"- **Invalidation Level (Stop Loss):** **{fm.price(inval)}** ({'200-EMA' if price > l['EMA200'] else '52-week low'})")
    A(f"- **Primary Bull Case Catalyst:** {bull}")
    A(f"- **Primary Bear Case Risk:** {', '.join(risks) if risks else 'Macro / sector de-rating (no model red flags)'}")
    A("\n> ⚠️ Auto-generated from free Yahoo Finance data for research purposes only - verify with company filings. Not investment advice.")

    META.update(sym=symbol, name=info.get("longName", symbol), sector=info.get("sector", ""),
                industry=info.get("industry", ""), country=info.get("country", ""), kpis=sector_kpis(info, F, fm))
    fig = make_chart(H, symbol) if show_charts else None
    return "\n".join(L), fig

# ---------------------------------------------------------------- chart
def make_chart(H, symbol, bars=250):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return None
    X = H.iloc[-bars:]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.15, 0.2, 0.15], vertical_spacing=0.02,
                        subplot_titles=(f"{symbol} Price, EMAs & Bollinger", "Volume", "MACD", "RSI (14)"))
    fig.add_trace(go.Candlestick(x=X.index, open=X.Open, high=X.High, low=X.Low, close=X.Close, name="Price"), 1, 1)
    for n, c in ((20, "orange"), (50, "royalblue"), (200, "firebrick")):
        fig.add_trace(go.Scatter(x=X.index, y=X[f"EMA{n}"], name=f"EMA{n}", line=dict(width=1.2, color=c)), 1, 1)
    fig.add_trace(go.Scatter(x=X.index, y=X.BB_UP, name="BB Up", line=dict(width=0.8, dash="dot", color="gray")), 1, 1)
    fig.add_trace(go.Scatter(x=X.index, y=X.BB_LO, name="BB Low", line=dict(width=0.8, dash="dot", color="gray")), 1, 1)
    fig.add_trace(go.Bar(x=X.index, y=X.Volume, name="Volume", marker_color="lightsteelblue"), 2, 1)
    fig.add_trace(go.Scatter(x=X.index, y=X.VOL_MA20, name="Vol MA20", line=dict(color="black", width=1)), 2, 1)
    fig.add_trace(go.Bar(x=X.index, y=X.MACD_HIST, name="Hist", marker_color="gray"), 3, 1)
    fig.add_trace(go.Scatter(x=X.index, y=X.MACD, name="MACD", line=dict(color="blue", width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=X.index, y=X.MACD_SIG, name="Signal", line=dict(color="red", width=1)), 3, 1)
    fig.add_trace(go.Scatter(x=X.index, y=X.RSI, name="RSI", line=dict(color="purple")), 4, 1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h"))
    return fig
