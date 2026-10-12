# Airport Investment Intelligence Agent

## Setup

Requires Python 3.11+.

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

Chat in the browser (type or record a question; allow microphone access for voice):

```bash
streamlit run app.py
```

Or chat in the terminal:

```bash
python -m airport_agent.cli                     # interactive chat (follow-up questions supported)
python -m airport_agent.cli "Your question"     # single question
```

## Test

```bash
pytest
```
