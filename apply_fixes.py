"""Usage: save your Colab code as engine.py (cell 2 only), then run: python apply_fixes.py"""
src = open("engine.py", encoding="utf-8").read()
def sub(old, new, count=None):
    global src
    n = src.count(old)
    assert n and (count is None or n == count), f"Pattern not found/unexpected count ({n}): {old[:60]}"
    src = src.replace(old, new)

# 1 dividend-yield unit bug (yfinance returns % in new versions; old *100 heuristic broke yields <1%)
sub('pct((info.get("dividendYield") or np.nan) * (1 if (info.get("dividendYield") or 0) > 1 else 100))', 'pct(div_yield(info))', 2)
sub('def cagr(first, last, years):', '''META = {}
def div_yield(info):
    v = info.get("trailingAnnualDividendYield")
    if isnum(v): return v * 100
    v = info.get("dividendYield")
    return v if isnum(v) else np.nan

def cagr(first, last, years):''', 1)
# 2 thesis text printed "(in)" for "In line"
sub("({status_vs(pe, R['hist_pe']).split(' ')[0].lower()})", "({'premium' if pe > R['hist_pe']*1.1 else 'discount' if pe < R['hist_pe']*0.9 else 'in line'})", 1)
# 3 CAGR mixed FY-2 -> TTM over a 2-year span; use FY-2 -> latest FY
sub('return dict(table=table, ratios=r, raw=raw)', 'return dict(table=table, ratios=r, raw=raw, fy={"Revenue": at(rev, 0), "EBITDA": at(ebitda, 0), "PAT": at(ni, 0)})', 1)
sub('cg = cagr(a, c, 2) if k in', 'cg = cagr(a, F["fy"].get(k, np.nan), 2) if k in', 1)
sub('cagr(F["table"]["Revenue"][0], F["table"]["Revenue"][2], 2)', 'cagr(F["table"]["Revenue"][0], F["fy"]["Revenue"], 2)')
sub("cagr(F['table']['Revenue'][0], F['table']['Revenue'][2], 2)", "cagr(F['table']['Revenue'][0], F['fy']['Revenue'], 2)")
# 4 FCF-based DCF is invalid for banks/NBFCs
sub('D = build_dcf(d, F, wacc, tg)\n', 'D = build_dcf(d, F, wacc, tg)\n    if info.get("sector") == "Financial Services":\n        D["fv"] = np.nan; D["note"] = "DCF not meaningful for banks/NBFCs - use P/B vs ROE"\n', 1)
# 5 terminal-growth grid could duplicate/negative
sub('tgs = [max(tg - 0.02, 0.0), tg - 0.01 if tg >= 0.01 else 0.0, tg, tg + 0.01, tg + 0.02]', 'tgs = [max(tg + k, 0.0) for k in (-0.02, -0.01, 0, 0.01, 0.02)]', 1)
# 6 expose metadata for the web layer
sub('    return "\\n".join(L), fig', '    META.update(sym=symbol, name=info.get("longName", symbol), sector=info.get("sector", ""), industry=info.get("industry", ""), country=info.get("country", ""), kpis=sector_kpis(info, F, fm))\n    return "\\n".join(L), fig', 1)
# 7 drop the notebook launcher
i = src.index('if __name__ == "__main__" or'); src = src[:i]
open("engine.py", "w", encoding="utf-8").write(src); print("engine.py patched")
