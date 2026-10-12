# Design — Airport Investment Intelligence Agent

<!-- Working draft: finalize against what the code actually does before shipping. Open items are marked TODO in comments. -->

## Overview

We rank US airports by how fast demand is outgrowing their runways and terminals, using a deterministic score built from public data, and an AI agent that explains the results — including what the data can't tell us.

```
Public APIs + 2 BTS downloads ──► ingest scripts ──► SQLite
                                                       │
                     deterministic metric & scoring functions (Python)
                                                       │
                     6 tools (each result carries source, data year, caveats)
                                                       │
            LangChain agent (Gemini by default, Groq switchable) ◄──► Streamlit chat
```

- **Rule:** the LLM never produces a number. Every metric, score, rank and difference comes from Python; the LLM chooses tools and explains their output.

## Scoring methodology

- **What the score measures:** need for added flight and passenger capacity — not financial return. The brief frames profit "based on increased flight and passenger capacity"; capacity need is the measurable precondition.
- **Score = weighted sum of four groups** (weights configurable by the analyst):

  | Group             | Default weight | Metrics                                                                   |
  | ----------------- | -------------- | ------------------------------------------------------------------------- |
  | Runway pressure   | 30             | Flight growth; flights per runway                                         |
  | Terminal pressure | 30             | Passenger growth; growth in passengers per flight                         |
  | Unmet demand      | 30             | Load factor; average fare                                                 |
  | Congestion        | 10             | Share of delays caused by the National Aviation System (capacity-related) |
- **Why these weights:** pressure on the airport itself matters most — two of three core groups are pressure (60 of 90 core points); unmet demand keeps a real share because an investor needs both. Congestion is low-weight because most delays are caused by weather or airlines, not capacity.
- **Metric definitions:**

  - Growth = annualized rate, 2019 → 2025 (skips COVID years 2020–22; full yearly series available to the analyst).
  - Flights per runway counts all departures including cargo (cargo uses runways too), over runways long enough for airline use. 
  - Passengers per flight counts passenger flights only. Rising passengers per flight = more people through the terminal per aircraft (bigger or fuller planes) — terminal pressure that runway metrics miss.
  - Average fare = passenger-weighted average of the airport's domestic airport-pair fares.
- **Normalization:** each metric → percentile rank (0–100) against all US airports; group score = mean of its metric percentiles. Raw values are always shown alongside.
- **Peer group:** national, so an airport's score doesn't change with the question; regional rank is shown too ("#1 of 9 in New England, 85th percentile nationally").
- **Size:** volume is not scored (we don't assume the investor prefers big or small airports). Airports under ~100k passengers/year are excluded (tiny bases produce meaningless growth rates); the analyst can filter by size in the chat.
- **Unmet demand is an estimate, not a measurement:** reported as an index (strength of signal, not a passenger count) plus a "who's the bottleneck?" label — full planes only signal a renovation opportunity when the airport is also under pressure ("airport-constrained" vs. "airline-side").
- **Other definitions:** long-haul = three distance bands (short < 800 mi, medium 800–3,000 mi, long > 3,000 mi; configurable), passenger flights only. Congestion is answered as a side-by-side profile (utilization, delays, delay causes, known constraints, live FAA status), not a single number.

<!-- TODO: add a bullet on missing data / outlier handling once the scoring code decides it. -->

## Key tradeoffs

- **Capacity need, not ROI.** We dropped an airport-finance metric: construction costs aren't public, and the brief defines profit through capacity.
- **Percentile rank over min–max scaling.** Robust to outlier airports and easy to explain, but near-ties can look far apart — mitigated by showing raw values.
- **No terminal-size data exists publicly.** Terminal pressure measures *growth* of pressure, not how full the terminal is today.
- **Full planes are a hint, not proof.** Airlines want full planes; the ranking still gives some credit for full planes at airports with spare room (the bottleneck label explains this; it doesn't fix it).
- **Endpoint growth (2019 → 2025).** Simple and skips COVID, but an unusual year at one airport distorts its growth.
- **APIs, with one stated exception.** Airport traffic, fares, runways, locations and live status come from public APIs (BTS, DOT, FAA). Route-level traffic and delay causes are only published as BTS downloads, so we use those files.
- **Data is a snapshot.** Data is fetched into SQLite by a script, with the data year shown on every answer. Known airport constraints (curfews, caps, slot rules) come from a small hand-curated, sourced and dated file covering a few airports — a missing entry doesn't mean no constraints.
- **LangChain over a plain SDK.** Provider switching (free-tier rate limits), built-in conversation memory and a recorded tool-call trace, at the cost of a hidden agent loop and a fast-changing library.
- **Conversations live in memory only.** A page refresh starts over; production would persist them for audit and resumption.
- **Out of scope (stated, not modelled):** airport finances, catchment population/income growth, competition from nearby airports, physical/regulatory room to expand, dependence on one airline, peak-hour congestion; Alaska and Hawaii have no fare data; US territories excluded.

## Where and how AI is used

### In the agent

- **The LLM does:** interpret the question, choose a tool and its arguments, map names to airport codes ("Santa Ana" → SNA), resolve follow-ups from conversation history ("why is it above Hartford?"), and write the explanation from tool output — including the caveats the tool returns.
- **The LLM does not:** produce any number, score, rank, difference or estimate; decide which airports are in a region; clean or join data.
- **Guardrails:** tools validate airport codes and suggest alternatives on unknown ones; every tool result carries source, data year and caveats; estimates are labelled as estimates; the system prompt forbids inventing numbers.
- **Model:** Gemini (free tier) by default; Groq switchable by one setting. Both reached through LangChain's prebuilt agent (`create_agent`) with six purpose-built tools.

### In building it

- I built this with Claude Code (Anthropic's AI coding assistant, Claude Opus 5.5) as a pair-programmer.
- **Design:** for each decision, Claude laid out 2–3 options with pros, cons and time cost; I made the call, and every decision and its reasoning is logged. Several key choices came from questioning its first suggestions — e.g., treating load factor as a hint rather than proof of unmet demand, and keeping the airport-vs-airline bottleneck distinction.
- **Data:** Claude tested the candidate public APIs live before we committed to them, which is how we found which sources have real APIs and which only offer downloads.
- **Code:** written in small increments, each explained before it was written and reviewed by me.
