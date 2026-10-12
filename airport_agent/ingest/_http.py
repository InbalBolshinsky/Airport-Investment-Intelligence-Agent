"""HTTP helpers shared by the ingest modules: one session, plus paging for the
two kinds of public API we use (Socrata open-data portals and ArcGIS)."""

import pandas as pd
import requests

TIMEOUT_SECONDS = 120

session = requests.Session()
session.headers["User-Agent"] = "airport-investment-agent (public-data research)"


def socrata_rows(
    domain: str,
    dataset_id: str,
    *,
    select: str | None = None,
    where: str | None = None,
    page_size: int = 50_000,
) -> pd.DataFrame:
    """Fetch every row of a Socrata dataset (data.bts.gov, datahub.transportation.gov).

    Pages through the API with $limit/$offset in a stable order (:id).
    All values arrive as strings; callers convert the columns they use.
    """
    url = f"https://{domain}/resource/{dataset_id}.json"
    params = {"$limit": page_size, "$order": ":id"}
    if select:
        params["$select"] = select
    if where:
        params["$where"] = where

    pages, offset = [], 0
    while True:
        response = session.get(url, params={**params, "$offset": offset}, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        pages.extend(page)
        offset += page_size
    return pd.DataFrame(pages)


def arcgis_rows(layer_url: str, *, where: str = "1=1", fields: str = "*", page_size: int = 2000) -> pd.DataFrame:
    """Fetch every record of an ArcGIS feature layer (FAA data hosted by DOT).

    Pages with resultOffset until the server stops reporting more records.
    Geometry is skipped; only attribute columns are returned.
    """
    records, offset = [], 0
    while True:
        response = session.get(
            f"{layer_url}/query",
            params={
                "where": where, "outFields": fields, "returnGeometry": "false", "f": "json",
                "resultOffset": offset, "resultRecordCount": page_size,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"ArcGIS error from {layer_url}: {body['error']}")
        features = body.get("features", [])
        records.extend(f["attributes"] for f in features)
        if not features or not body.get("exceededTransferLimit"):
            break
        offset += len(features)
    return pd.DataFrame(records)
