"""Setup check: missing or partly loaded databases are reported, and checking never creates a file."""

import dataclasses
import sqlite3
from contextlib import closing

import pytest

from airport_agent import db


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Point the project settings at a database file in a temporary folder."""
    path = tmp_path / "airports.db"
    monkeypatch.setattr(db, "settings", dataclasses.replace(db.settings, db_path=path))
    return path


def create_tables(path, tables) -> None:
    with closing(sqlite3.connect(path)) as conn:
        for table in tables:
            conn.execute(f"CREATE TABLE {table} (x INTEGER)")
        conn.commit()


def test_no_database_file(db_path):
    assert db.missing_tables() == list(db.REQUIRED_TABLES)
    assert "No data yet" in db.data_problem()
    assert not db_path.exists()  # the check must not create an empty database


def test_failed_source_is_named_with_its_rerun_command(db_path):
    create_tables(db_path, [t for t in db.REQUIRED_TABLES if t != "routes"])
    assert db.missing_tables() == ["routes"]
    assert "python -m airport_agent.ingest bts_routes" in db.data_problem()


def test_complete_database_has_no_problem(db_path):
    create_tables(db_path, db.REQUIRED_TABLES)
    assert db.data_problem() is None
