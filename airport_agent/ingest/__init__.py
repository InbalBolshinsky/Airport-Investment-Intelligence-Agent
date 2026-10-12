"""Data ingestion: fetch each public source, clean it lightly, load it into SQLite.

One module per source. Run all of them with `python -m airport_agent.ingest`,
or some of them by name, e.g. `python -m airport_agent.ingest regions`.
"""
