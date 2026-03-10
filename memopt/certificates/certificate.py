"""
OptimizationCertificate — signed audit record per committed optimization.

Each time OptimizationExecutor commits a speedup (decision == "COMMIT"),
a certificate is issued and stored in ~/.memopt/certificates.db.

Fields:
    certificate_id      — UUID4 hex
    issued_at           — Unix timestamp
    model_name          — caller-supplied label (default "unknown")
    optimization_type   — e.g. "torch_compile", "channels_last"
    baseline_ms         — median inference time before optimization
    optimized_ms        — median inference time after optimization
    speedup_pct         — ((baseline - optimized) / baseline) × 100
    correctness_validated — True when correctness check passed
    device              — "cuda:0" / "cpu"
    memopt_version      — package version string
    signature           — HMAC-SHA256 of key fields using stored secret

Usage:
    store = CertificateStore()
    cert  = store.issue(result, model_name="bert-base")
    certs = store.list_certs(limit=20)
    ok    = store.verify(cert)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("memopt.certificates")

_MEMOPT_VERSION = "1.0.0"
_DB_DEFAULT = str(Path.home() / ".memopt" / "certificates.db")


# ── Certificate dataclass ────────────────────────────────────────────────────

@dataclass
class OptimizationCertificate:
    certificate_id:       str
    issued_at:            float
    model_name:           str
    optimization_type:    str
    baseline_ms:          float
    optimized_ms:         float
    speedup_pct:          float
    correctness_validated: bool
    device:               str
    memopt_version:       str
    signature:            str

    def as_dict(self) -> dict:
        return asdict(self)


# ── Certificate store ────────────────────────────────────────────────────────

class CertificateStore:
    """
    SQLite-backed store for OptimizationCertificate records.

    Thread-safe for concurrent readers; single writer at a time
    (SQLite WAL mode).
    """

    def __init__(self, db_path: str = _DB_DEFAULT) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── private ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS certificates (
                    certificate_id       TEXT PRIMARY KEY,
                    issued_at            REAL NOT NULL,
                    model_name           TEXT NOT NULL,
                    optimization_type    TEXT NOT NULL,
                    baseline_ms          REAL NOT NULL,
                    optimized_ms         REAL NOT NULL,
                    speedup_pct          REAL NOT NULL,
                    correctness_validated INTEGER NOT NULL,
                    device               TEXT NOT NULL,
                    memopt_version       TEXT NOT NULL,
                    signature            TEXT NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cert_issued_at "
                "ON certificates(issued_at DESC)"
            )

    @staticmethod
    def _signing_secret() -> bytes:
        """
        Per-install HMAC secret.

        Uses the memopt API key (stored in ~/.memopt/api_key) as the
        signing material.  Falls back to a deterministic but node-unique
        value derived from the DB path so that certificates remain
        verifiable on reinstall.
        """
        try:
            from memopt.auth.api_key import get_or_create_key
            return get_or_create_key().encode()
        except Exception:
            return hashlib.sha256(_DB_DEFAULT.encode()).digest()

    @staticmethod
    def _sign(cert_id: str, opt_type: str, baseline: float,
              optimized: float, speedup: float, secret: bytes) -> str:
        """HMAC-SHA256 over the numeric fields that define the certificate."""
        payload = f"{cert_id}|{opt_type}|{baseline:.6f}|{optimized:.6f}|{speedup:.6f}"
        return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()

    # ── public API ───────────────────────────────────────────────────────────

    def issue(
        self,
        result,                          # OptimizationResult
        model_name: str = "unknown",
        device: str = "",
    ) -> OptimizationCertificate:
        """
        Issue and persist a certificate for a committed optimization.

        Raises ValueError if result.success is False.
        """
        if not result.success:
            raise ValueError(
                "Cannot issue certificate for a non-committed result "
                f"(optimization_type={result.optimization_type!r})"
            )

        if not device:
            try:
                import torch
                device = str(next(iter(
                    p.device for p in []
                )))
            except Exception:
                pass
            if not device:
                try:
                    import torch
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"

        cert_id = uuid.uuid4().hex
        secret  = self._signing_secret()
        sig     = self._sign(
            cert_id,
            result.optimization_type,
            result.baseline_time_ms,
            result.optimized_time_ms,
            result.speedup_pct,
            secret,
        )

        cert = OptimizationCertificate(
            certificate_id=cert_id,
            issued_at=time.time(),
            model_name=model_name,
            optimization_type=result.optimization_type,
            baseline_ms=result.baseline_time_ms,
            optimized_ms=result.optimized_time_ms,
            speedup_pct=result.speedup_pct,
            correctness_validated=bool(result.correctness_validated),
            device=device,
            memopt_version=_MEMOPT_VERSION,
            signature=sig,
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO certificates VALUES
                (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                cert.certificate_id,
                cert.issued_at,
                cert.model_name,
                cert.optimization_type,
                cert.baseline_ms,
                cert.optimized_ms,
                cert.speedup_pct,
                int(cert.correctness_validated),
                cert.device,
                cert.memopt_version,
                cert.signature,
            ))

        log.info(
            "Certificate issued: %s | %s | %.1f%% speedup",
            cert_id[:8], cert.optimization_type, cert.speedup_pct,
        )
        return cert

    def verify(self, cert: OptimizationCertificate) -> bool:
        """Return True if the certificate signature is valid."""
        secret   = self._signing_secret()
        expected = self._sign(
            cert.certificate_id,
            cert.optimization_type,
            cert.baseline_ms,
            cert.optimized_ms,
            cert.speedup_pct,
            secret,
        )
        return hmac.compare_digest(cert.signature, expected)

    def list_certs(
        self,
        limit: int = 50,
        model_name: Optional[str] = None,
        optimization_type: Optional[str] = None,
    ) -> List[OptimizationCertificate]:
        """Return certificates newest-first."""
        query = "SELECT * FROM certificates WHERE 1=1"
        params: list = []
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        if optimization_type:
            query += " AND optimization_type = ?"
            params.append(optimization_type)
        query += " ORDER BY issued_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            OptimizationCertificate(
                certificate_id=r[0], issued_at=r[1], model_name=r[2],
                optimization_type=r[3], baseline_ms=r[4], optimized_ms=r[5],
                speedup_pct=r[6], correctness_validated=bool(r[7]),
                device=r[8], memopt_version=r[9], signature=r[10],
            )
            for r in rows
        ]

    def get(self, certificate_id: str) -> Optional[OptimizationCertificate]:
        """Look up a single certificate by ID prefix or full ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM certificates WHERE certificate_id LIKE ?",
                (certificate_id + "%",),
            ).fetchone()
        if not row:
            return None
        return OptimizationCertificate(
            certificate_id=row[0], issued_at=row[1], model_name=row[2],
            optimization_type=row[3], baseline_ms=row[4], optimized_ms=row[5],
            speedup_pct=row[6], correctness_validated=bool(row[7]),
            device=row[8], memopt_version=row[9], signature=row[10],
        )

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM certificates"
            ).fetchone()[0]
