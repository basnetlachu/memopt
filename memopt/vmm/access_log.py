"""
Access logging — non-blocking, append-only JSONL event log for block accesses.

Every block access is recorded as a BlockAccessEvent and written to a
rotating daily JSONL file via a background daemon thread.  The caller is
never blocked — events are enqueued and dropped silently when the queue
is full.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockAccessEvent:
    sequence_id: str
    block_index: int
    token_position: int         # which generation step triggered this access
    attention_layer: int        # which transformer layer (-1 if unknown)
    timestamp: float            # time.perf_counter()
    tier_at_access: str         # "hbm" | "dram" | "nvme" | "unknown"
    promotion_latency_ms: float # 0.0 if already in HBM, actual ms if promoted
    tenant_id: str


class AccessLog:
    """Thread-safe, non-blocking JSONL access logger with background writer."""

    _QUEUE_MAXSIZE = 10_000
    _SENTINEL = object()

    def __init__(self, log_dir: str | None = None) -> None:
        self._events_recorded = 0
        self._events_dropped = 0
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._file = None
        self._log_file_path: str = ""

        try:
            if log_dir is None:
                log_dir = os.path.join(Path.home(), ".memopt", "access_logs")
            os.makedirs(log_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            self._log_file_path = os.path.join(log_dir, f"access_log_{date_str}.jsonl")
            self._file = open(self._log_file_path, "a", encoding="utf-8")
        except Exception:
            logger.debug("AccessLog: failed to open log file", exc_info=True)
            self._file = None

        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="access-log-writer"
        )
        self._writer_thread.start()

    def record(self, event: BlockAccessEvent) -> None:
        """Enqueue an event for background writing.  Never blocks the caller."""
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            with self._lock:
                self._events_dropped += 1

    def stats(self) -> dict:
        with self._lock:
            size = 0
            try:
                if self._log_file_path and os.path.exists(self._log_file_path):
                    size = os.path.getsize(self._log_file_path)
            except OSError:
                pass
            return {
                "events_recorded": self._events_recorded,
                "events_dropped": self._events_dropped,
                "log_file_path": self._log_file_path,
                "log_size_bytes": size,
            }

    def shutdown(self) -> None:
        """Flush remaining events and close the file handle."""
        try:
            self._queue.put(self._SENTINEL, timeout=5.0)
        except queue.Full:
            pass
        self._writer_thread.join(timeout=10.0)
        try:
            if self._file and not self._file.closed:
                self._file.close()
        except Exception:
            logger.debug("AccessLog: error closing file", exc_info=True)

    # ── background writer ─────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is self._SENTINEL:
                # drain anything remaining
                self._drain()
                return
            self._write_event(item)

    def _drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is self._SENTINEL:
                continue
            self._write_event(item)
        try:
            if self._file and not self._file.closed:
                self._file.flush()
        except Exception:
            logger.debug("AccessLog: flush error", exc_info=True)

    def _write_event(self, event: BlockAccessEvent) -> None:
        try:
            if self._file and not self._file.closed:
                line = json.dumps(dataclasses.asdict(event))
                self._file.write(line + "\n")
                self._file.flush()
                with self._lock:
                    self._events_recorded += 1
        except Exception:
            logger.debug("AccessLog: write error", exc_info=True)
