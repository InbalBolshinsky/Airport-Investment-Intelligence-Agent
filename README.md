# Airport Investment Intelligence Agent

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add a free API key:

- **Gemini** (default): get a key at https://aistudio.google.com and set `GOOGLE_API_KEY`.
- **Groq** (alternative): get a key at https://console.groq.com, set `GROQ_API_KEY` and `LLM_PROVIDER=groq`.

## Load the data

Fetches public BTS, DOT and FAA data into `data/airports.db` (about 3 minutes; no API key needed):

```bash
python -m airport_agent.ingest
```

## Run

```bash
python -m airport_agent.cli                     # interactive chat (follow-up questions supported)
python -m airport_agent.cli "Your question"     # single question
```
