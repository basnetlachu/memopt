"""
memopt control plane — centralized cluster management.

Components:
  database.py   SQLite schema + queries (nodes, events, metrics)
  server.py     FastAPI server + all endpoints
  dashboard.html Single-file HTML dashboard
  cli.py        memopt cluster status/nodes/events commands
"""
from .database import Database, NodeRecord, EventRecord
from .server import app

__all__ = ["Database", "NodeRecord", "EventRecord", "app"]
