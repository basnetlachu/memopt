"""
Oracle Trainer — background daemon that warms a MemoryOracle from
existing access logs and incrementally trains on new events.

Uses OracleDataCleaner for full 7-step cleaning on every batch.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import tempfile
import threading
from typing import TYPE_CHECKING

from .oracle_data_cleaner import OracleDataCleaner

if TYPE_CHECKING:
    from .oracle import MemoryOracle

logger = logging.getLogger(__name__)


class OracleTrainer:

    def __init__(
        self,
        oracle: "MemoryOracle",
        log_dir: str,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._oracle = oracle
        self._log_dir = log_dir
        self._poll_interval = poll_interval_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._file_positions: dict[str, int] = {}
        self._live_buffer: dict[str, list[str]] = {}
        self._events_trained = 0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="oracle-trainer",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def stats(self) -> dict:
        with self._lock:
            return {
                "events_trained": self._events_trained,
                "files_watched": len(self._file_positions),
                "log_dir": self._log_dir,
                "poll_interval_s": self._poll_interval,
                "oracle_stats": self._oracle.stats().__dict__,
            }

    # ── Internal ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            self._warm_from_existing_logs()
        except Exception as exc:
            logger.debug("OracleTrainer warm error: %s", exc)
        self._poll_loop()

    def _warm_from_existing_logs(self) -> None:
        try:
            files = sorted(
                glob.glob(os.path.join(self._log_dir, "*.jsonl"))
            )
        except OSError:
            return

        for path in files:
            try:
                cleaner = OracleDataCleaner()
                clean_events, st = cleaner.clean_file(path)
                logger.info(cleaner.stats_summary(st))
                for ev in clean_events:
                    self._oracle.observe(
                        sequence_id=ev["sequence_id"],
                        block_index=ev["block_index"],
                        step=ev.get("token_position"),
                    )
                    with self._lock:
                        self._events_trained += 1
                filename = os.path.basename(path)
                self._file_positions[filename] = os.path.getsize(path)
            except Exception as exc:
                logger.debug("OracleTrainer warm file error: %s", exc)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._ingest_new_events()
            except Exception as exc:
                logger.debug("OracleTrainer poll error: %s", exc)

    def _ingest_new_events(self) -> None:
        # Phase 1 — Buffer new raw lines per file
        try:
            files = glob.glob(os.path.join(self._log_dir, "*.jsonl"))
        except OSError:
            return

        for path in files:
            filename = os.path.basename(path)
            offset = self._file_positions.get(filename, 0)
            try:
                with open(path, "r") as f:
                    f.seek(offset)
                    new_lines = f.readlines()
                    if new_lines:
                        self._live_buffer.setdefault(filename, []).extend(
                            new_lines,
                        )
                    self._file_positions[filename] = f.tell()
            except (OSError, IOError):
                continue

        # Phase 2 — Full clean on buffered lines
        all_lines: list[str] = []
        for lines in self._live_buffer.values():
            all_lines.extend(lines)

        if not all_lines:
            return

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".jsonl")
            with os.fdopen(fd, "w") as f:
                for line in all_lines:
                    f.write(line if line.endswith("\n") else line + "\n")

            cleaner = OracleDataCleaner()
            clean_events, st = cleaner.clean_file(tmp_path)
            logger.debug(cleaner.stats_summary(st))

            for ev in clean_events:
                self._oracle.observe(
                    sequence_id=ev["sequence_id"],
                    block_index=ev["block_index"],
                    step=ev.get("token_position"),
                )
                with self._lock:
                    self._events_trained += 1

            # Retain lines from sequences not in clean output
            # (they may reach min_sequence_length on the next poll)
            clean_seq_ids = {e["sequence_id"] for e in clean_events}
            retained: dict[str, list[str]] = {}
            for filename, lines in self._live_buffer.items():
                kept: list[str] = []
                for line in lines:
                    try:
                        data = json.loads(line)
                        sid = data.get("sequence_id", "")
                        if sid and sid not in clean_seq_ids:
                            kept.append(line)
                    except Exception:
                        pass
                if kept:
                    retained[filename] = kept
            self._live_buffer = retained

        except Exception as exc:
            logger.debug("OracleTrainer ingest error: %s", exc)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
