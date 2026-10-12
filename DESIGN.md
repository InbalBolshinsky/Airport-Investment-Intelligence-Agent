# Design - Airport Investment Intelligence Agent

## Overview

Ranking US airports by how fast demand is outgrowing their capacity, using a deterministic score over public data, with an AI agent that explains the results and their limits.

```
Public APIs + BTS downloads ─► SQLite ─► metrics & scoring (Python) ─► 6 tools ─► LLM agent ─► chat (text/voice)
```

- **The LLM** understands the question, picks a tool and its inputs, handles follow-ups, and writes the answer from the tool's results and caveats. It never produces or calculates a number: every metric, score, rank and comparison is calculated by deterministic Python code, and a "How I got this" panel shows the tool calls and raw results behind each answer.
- **Models:** Gemini 3.6 Flash (free tier), falling back to Gemini 3.5 Flash-Lite and then Groq if one is overloaded or out of quota. Spoken questions are transcribed by Groq Whisper (Gemini as backup).

## Scoring methodology

- **What we measure:** need for more flight and passenger capacity. We prefer not to measure financial return.
- **Score (0–100):** a weighted average of four groups. The analyst can change the weights by asking in the chat (e.g. "rank again without congestion"); this applies to rankings and comparisons, while single-airport answers use the defaults and say so.

  | Group             | Weight | Metrics                                                                                                     |
  | ----------------- | ------ | ----------------------------------------------------------------------------------------------------------- |
  | Runway pressure   | 30     | Flights per long runway (counts 2×); flight growth                                                         |
  | Terminal pressure | 30     | Passenger growth; growth in passengers per flight                                                           |
  | Unmet demand      | 30     | Load factor; fare premium vs. similar-distance trips                                                        |
  | Congestion        | 10     | Share of arriving flights delayed by the aviation system (traffic, airport operations, non-extreme weather) |
- **Why these weights:** pressure on the airport itself matters most, but an investor also needs signs of unmet demand. Congestion counts little because most delays come from weather and airlines. Within runway pressure, a runway that is busy today counts twice as much as flight growth: a full airport is a surer bet than one growing fast from a small base.
- **Definitions:**

  - Growth: yearly average, 2019 → 2025 (skips the COVID years).
  - Flights per long runway: all 2025 departures, cargo included, per runway of 5,000 ft or more.
  - Passengers per flight: passenger planes only. When it rises, more people pass through the terminal per flight.
  - Fare premium: how far fares sit above the national typical fare for trips of the same distance.
  - Long-haul: over 3,000 miles (short < 800, medium in between; adjustable), for passenger flights and for all flights incl. cargo.
- **Percentiles:** each metric becomes a percentile (0–100) against all 226 ranked US airports (100k+ departing passengers in 2025). They are computed nationally, so a score never changes with the question asked.
- **Size isn't scored**, because we don't know whether the investor prefers big or small airports. Answers show each airport's size and the analyst can filter by it. Counts are departing passengers, about half of what airports usually report.
- **Bottleneck label:** "airport-constrained" means planes are full *and* the airport is among the busiest today by flights per long runway (both ≥ 70th percentile, i.e. top 30%: clearly above average, not rare). Full planes at an airport with room are "airline-side": airlines could add flights themselves. "Busy" is relative to other airports; true capacity isn't public. Delays are left out of the label: one month of them is noisy at small airports and includes weather (a foggy 135k-passenger airport ranked in the top 1% for delays).
- **Strong candidate:** high score (≥ 70th percentile) *and* airport-constrained. Computed in code.
- **Unmet demand** is an estimate of people who would fly from an airport but can't get a seat. It can't be counted, so we infer it from two signals, averaged as percentiles: full planes (high load factor: seats sell out) and high fares for the distance (scarce seats push prices up). The result is an index of how strong the signals are, not a count of passengers. Where fares are missing, it rests on load factor alone.
- **Missing data:** an airport is scored on what it has and the answer lists what's missing. Nothing is filled in. If custom weights leave an airport with no data in any weighted group, it gets no score or rank, and the answer says why.
- **Outliers:** percentiles cap extreme values. Growth from a tiny 2019 base keeps its rank but carries a warning that it's likely a one-time jump (e.g. a new airline base).

## Key tradeoffs

- **Need, not ROI.** Construction costs aren't public, so we score capacity need only.
- **Percentiles, not raw scaling.** One extreme airport can't distort everyone else, but near-ties can look far apart, so raw values are shown too.
- **A hard cut-off for "strong candidate".** The label needs a score in the top 30%, so airports near the line flip when the weights change: without the congestion group (one month of delays), Boston drops from the 73rd to the 63rd percentile and loses it. Only Dallas/Fort Worth, Newark and Greenville-Spartanburg keep it whichever group is removed.
- **No terminal-size data exists publicly.** Terminal pressure measures how fast pressure grows, not how full a terminal is. So for terminal questions the strong-candidate flag rests on busy runways and full planes: Boston, New England's only strong candidate, ranks 8th of 10 there on terminal pressure. Answers show that group's score.
- **Rankings lean toward small, fast-growing airports.** Big hubs are busy but many are still below 2019 traffic, partly because growth at a busy airport is limited by its capacity. Growth also enters three times (flights, passengers, passengers per flight; the first two move together), so across the score it carries about twice the weight of how busy an airport is today. We kept size unscored and added the strong-candidate flag, so busy hubs aren't judged on growth alone. In New England only Boston is a strong candidate; Bangor and Portland score higher but have room on their runways.
- **Full planes are a hint, not proof.** Airlines want full planes. The score still rewards them a little at airports with room; the bottleneck label points this out. The reverse also happens: Santa Ana and Reagan National are legally capped, but their planes aren't unusually full, so their label shows no unmet-demand signal. Known caps are shown in answers (Santa Ana's is in the hand-checked list) but don't change the label.
- **Growth from two endpoints (2019 vs 2025).** Simple, but one unusual year distorts it.
- **APIs, with exceptions.** Traffic, fares, runways and locations come from BTS, DOT and FAA APIs. Routes and delay causes are only published as downloads. Delays cover the latest single month (a full year takes ~2 hours to download), so they're seasonal. The data is a snapshot; one command rebuilds it in ~3 minutes.
- **Partial data.** Fares cover 143 of 226 airports and none in Alaska or Hawaii. Known constraints (caps, curfews, runway spacing) are hand-checked, sourced and dated for SNA, SFO and LAX only.
- **Two data gaps the code doesn't guard against** (neither occurs in today's data). An airport with no runway of 5,000 ft or more in the FAA data would have its full planes labelled "airline-side" (room on the runways) when its runway use is actually unknown. And the delay download is assumed to hold one month; if BTS ever returned several, airports would be counted more than once.
- **LangChain, not a plain SDK.** It gives easy model switching, fallbacks and conversation memory, but hides the agent loop and changes often. Conversations live in memory only: a refresh or restart starts a new one. Each turn resends the whole conversation, which suits short analyst sessions (from about turn 6, older tool results are trimmed to save quota); the Groq fallback's free-tier limit means it mainly helps early in a conversation.
- **Out of scope:** airport finances, population growth, competition between nearby airports, room to expand.

## Where and how AI is used

I built this with Claude Code (Claude Opus 5.5) as a pair-programmer, working phase by phase from design to verification.

- **Design:** for every decision, Claude laid out multiple options with pros, cons and time cost; I chose, and kept a private decision log. Several key choices came from questioning its suggestions - e.g. treating full planes as a hint rather than proof of unmet demand, separating airport from airline bottlenecks, and correcting "a full airport can't grow" to "growth is limited by capacity".
- **Data:** Claude tested every candidate API live before we relied on it, which showed which sources have real APIs and which only offer downloads, and surfaced data issues we then handled (e.g. passenger counts are departing only; one airport missing from the FAA list).
- **Code:** written in small increments, each explained before it was written and reviewed by me, with unit tests for the scoring.
- **Checks:** Claude ran the example questions through the agent, sanity-checked rankings with me (which led to fixes such as SFO's bottleneck label), and verified a clean install from the README.
