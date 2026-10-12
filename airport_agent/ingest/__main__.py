"""Runner: `python -m airport_agent.ingest [source ...]`.

Runs every source (or the ones named) into the project database. A source
that fails is reported and the others still run; the exit code is 1 if any
source failed.
"""

import sys
import time

from airport_agent.config import settings
from airport_agent.db import connect
from airport_agent.ingest import bts_delays, bts_routes, bts_traffic, dot_fares, faa, regions

# Order matters only for readability of the log; sources are independent.
SOURCES = {
    "regions": regions.ingest,
    "bts_traffic": bts_traffic.ingest,
    "faa": faa.ingest,
    "bts_routes": bts_routes.ingest,
    "dot_fares": dot_fares.ingest,
    "bts_delays": bts_delays.ingest,
}


def main(names: list[str]) -> int:
    """Run the named sources (all if none given). Returns a process exit code."""
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        print(f"Unknown source(s): {unknown}. Available: {list(SOURCES)}")
        return 1

    print(f"Loading into {settings.db_path}")
    failures = []
    with connect() as conn:
        for name in names or list(SOURCES):
            start = time.time()
            try:
                rows = SOURCES[name](conn)
                print(f"  {name:12} {rows:>9,} rows  ({time.time() - start:.0f}s)")
            except Exception as error:  # report and keep going with the other sources
                failures.append(name)
                print(f"  {name:12} FAILED: {type(error).__name__}: {error}")

    if failures:
        print(f"Failed: {failures}. Re-run them with: python -m airport_agent.ingest {' '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
