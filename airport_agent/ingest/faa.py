"""US airports and their runways, from FAA data published by DOT (ArcGIS APIs).

Sources: FAA NASR airport and runway records, in DOT's National Transportation
Atlas Database (NTAD), served as ArcGIS feature layers.

Cleaning:
- Airports: keep landing facilities of type "A" (airports; not heliports or
  seaplane bases), with code, name, city, state and coordinates.
- Airports get a `bts_code` column for joining with BTS traffic: the same as
  the FAA code, except for the few listed in reference/airport_codes.csv.
- Runways: keep every runway of those airports with its length. The minimum
  length that counts toward capacity is applied in the metric code, not here.

Known gaps:
- Palm Beach International (PBI) is missing from the FAA dataset, so it has
  no state or runway count and is left out of regional rankings.
- Runway length alone does not say which aircraft can use a runway.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from airport_agent.db import record_source, replace_table
from airport_agent.ingest._http import arcgis_rows

ARCGIS_BASE = "https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services"
AIRPORTS_URL = f"{ARCGIS_BASE}/NTAD_Aviation_Facilities/FeatureServer/0"
RUNWAYS_URL = f"{ARCGIS_BASE}/Runways_View/FeatureServer/0"
CODE_TRANSLATIONS = Path(__file__).resolve().parent.parent / "reference" / "airport_codes.csv"


def ingest_airports(conn: sqlite3.Connection) -> int:
    """Load `airports`: one row per US airport (FAA code, name, city, state, coordinates)."""
    raw = arcgis_rows(
        AIRPORTS_URL,
        where="SITE_TYPE_CODE = 'A'",
        fields="ARPT_ID,ARPT_NAME,CITY,STATE_CODE,LAT_DECIMAL,LONG_DECIMAL,EFF_DATE",
    )
    df = raw.rename(columns={
        "ARPT_ID": "airport", "ARPT_NAME": "airport_name", "CITY": "city",
        "STATE_CODE": "state", "LAT_DECIMAL": "lat", "LONG_DECIMAL": "lon",
    })
    effective = str(df.pop("EFF_DATE").max())
    df = df.drop_duplicates("airport")

    faa_to_bts = pd.read_csv(CODE_TRANSLATIONS).set_index("faa_code")["bts_code"]
    df["bts_code"] = df["airport"].map(faa_to_bts).fillna(df["airport"])

    count = replace_table(conn, "airports", df)
    record_source(
        conn,
        table="airports",
        source="FAA airport records via DOT NTAD Aviation Facilities (ArcGIS API)",
        url=AIRPORTS_URL,
        period=f"FAA data effective {effective}",
        row_count=count,
        known_gaps=(
            "`airport` is the FAA identifier; join BTS data on `bts_code` (differs for the airports in "
            "reference/airport_codes.csv). PBI (Palm Beach) is missing from the FAA dataset."
        ),
    )
    return count


def ingest_runways(conn: sqlite3.Connection) -> int:
    """Load `runways`: one row per runway (airport code, runway id, length in feet)."""
    raw = arcgis_rows(
        RUNWAYS_URL,
        where="SITE_TYPE_CODE = 'A'",
        fields="ARPT_ID,RWY_ID,RWY_LEN,SURFACE_TYPE_CODE,EFF_DATE",
    )
    df = raw.rename(columns={
        "ARPT_ID": "airport", "RWY_ID": "runway", "RWY_LEN": "length_ft", "SURFACE_TYPE_CODE": "surface",
    })
    effective = str(df.pop("EFF_DATE").max())
    df["length_ft"] = pd.to_numeric(df["length_ft"], errors="coerce")

    count = replace_table(conn, "runways", df)
    record_source(
        conn,
        table="runways",
        source="FAA runway records via DOT NTAD Runways (ArcGIS API)",
        url=RUNWAYS_URL,
        period=f"FAA data effective {effective}",
        row_count=count,
        known_gaps="Length only; does not say which aircraft a runway supports.",
    )
    return count


def ingest(conn: sqlite3.Connection) -> int:
    """Load both FAA tables. Returns total rows."""
    return ingest_airports(conn) + ingest_runways(conn)
