import os, datetime as dt, functools, anthropic
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
   def _client():
       return anthropic.Anthropic()  # needs ANTHROPIC_API_KEY
TOOLS = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}]
RULES = ("You are an equity research analyst. Use web search for current facts. Cite source name and date inline. "
         "Never invent numbers, price targets, quotes or names; if something is not found write 'Not found in public sources'. "
         "Paraphrase, do not copy long passages. Output concise Markdown (tables where useful), no preamble. Educational, not investment advice.")
P = {
"industry": "Where does the {industry} industry (sector: {sector}) stand today? Give: (1) GLOBAL: market size/growth, cycle position, key trends, leaders; (2) {country} / India-level: size, growth, policy and regulatory tailwinds/headwinds, competitive structure; (3) where {name} sits in it (market share/rank) and a one-line verdict: expanding, mature or cyclical downturn.",
"business": "Explain the business model of {name} ({symbol}) in depth: what it sells, segments with revenue mix %, customers, geographies, how it makes money (pricing, unit economics, recurring vs one-off), value chain position, competitive moat (and its durability), capital intensity, key cost drivers, and main risks to the model.",
"research": "Find the most recent broker/fund-house research on {name} ({symbol}) (last ~6 months): a table of Firm | Date | Rating | Target price | Key rationale. Then summarise consensus view, range of targets, and where analysts disagree. Only include reports you actually found.",
"concall": "Analyse the most recent earnings concall (and results) of {name} ({symbol}): date/quarter, management guidance (revenue, margin, capex, order book), key positives, concerns and evasive/unanswered analyst questions, changes in tone vs the prior call, and 3 things to track next quarter.",
"management": "Profile the leadership of {name} ({symbol}): table of Name | Role | Tenure | Background | Notable track record. Then assess competence and governance: execution vs past guidance, capital allocation, promoter/insider holding and pledging, related-party dealings, auditor changes, remuneration, board independence, any controversies or regulatory actions. Give a clearly labelled analyst-judgement rating (Strong/Adequate/Weak) with evidence.",
}
@functools.lru_cache(maxsize=256)
def _run(section, name, symbol, sector, industry, country, day):
    prompt = P[section].format(name=name, symbol=symbol, sector=sector or "n/a", industry=industry or "n/a", country=country or "India")
    r = _client().messages.create((model=MODEL, max_tokens=3000, system=RULES, tools=TOOLS, messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text").strip()
def run(section, **k):
    return _run(section, k["name"], k["symbol"], k["sector"], k["industry"], k["country"], dt.date.today().isoformat())
