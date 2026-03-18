"""
Audit Logger
-------------
Append-only JSONL audit log with HMAC-SHA256 signed entries.

Each entry includes:
  - A `prev_hash` that chains it to the previous entry (tamper-evidence).
  - An `hmac_sig` computed over the entry bytes + prev_hash.

The HMAC secret is read from the AUDIT_HMAC_KEY environment variable.
A default dev key is used if the variable is absent (logs a warning).

Usage:
    logger = AuditLogger()
    logger.append(AuditEntry(action="read_billing", ...))
    ok, breaks = logger.verify_chain()
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    """One audit log record."""

    action: str
    allowed: bool
    user_role: str = "anonymous"
    data_classification: str = "internal"
    username: str = ""
    query_hash: str = ""          # SHA-256 of raw query text (no PII stored)
    policy_ids_evaluated: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    timestamp: str = ""           # set automatically in AuditLogger.append()

    # Set by logger (not caller):
    prev_hash: str = ""
    hmac_sig:  str = ""


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_DEFAULT_LOG_PATH = Path("data/audit/audit.jsonl")
_DEV_HMAC_KEY     = b"dev-only-insecure-key-change-in-prod"
_GENESIS_HASH     = "genesis"


class AuditLogger:
    """
    Append-only JSONL audit logger with HMAC chain integrity.

    Thread-safety: appends are serialised via a simple in-process approach.
    For multi-process deployments, use a write-once blob store instead.
    """

    def __init__(self, log_path: str | Path = _DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        raw_key = os.environ.get("AUDIT_HMAC_KEY", "")
        if not raw_key:
            env = os.environ.get("ENVIRONMENT", "dev")
            if env == "production":
                raise RuntimeError(
                    "[AuditLogger] AUDIT_HMAC_KEY must be set in production. "
                    "Set ENVIRONMENT=dev to use the insecure dev key."
                )
            _logger.warning(
                "[AuditLogger] AUDIT_HMAC_KEY not set in environment. "
                "Using insecure dev key -- do not use in production."
            )
            self._key = _DEV_HMAC_KEY
        else:
            self._key = raw_key.encode()

        # Cache the hash of the last entry for chaining
        self._last_hash: str = self._read_last_hash()
        _logger.debug(
            f"[AuditLogger] Initialised | path={self.log_path} | "
            f"last_hash={self._last_hash[:16]}..."
        )

    # -------------------------------------------------------------------------
    # Append
    # -------------------------------------------------------------------------

    def append(self, entry: AuditEntry) -> None:
        """Sign and append an entry to the log. Sets timestamp, prev_hash, hmac_sig."""
        entry.timestamp = datetime.now(timezone.utc).isoformat()
        entry.prev_hash = self._last_hash

        # Compute HMAC over canonical JSON (sorted keys, no sig field)
        entry_dict = asdict(entry)
        entry_dict.pop("hmac_sig", None)
        canonical = json.dumps(entry_dict, sort_keys=True, ensure_ascii=False)
        entry.hmac_sig = self._sign(canonical.encode())

        # Write line (newline="" forces \n on all platforms, preventing \r\n on Windows)
        line = json.dumps(asdict(entry), ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")

        # Update chain hash
        self._last_hash = self._hash_bytes(line.encode())

        _logger.debug(
            f"[AuditLogger] APPEND | action={entry.action} "
            f"allowed={entry.allowed} user={entry.username}"
        )

    # -------------------------------------------------------------------------
    # Verify chain
    # -------------------------------------------------------------------------

    def verify_chain(self) -> tuple[bool, list[int]]:
        """
        Walk the entire log and verify every HMAC and prev_hash chain link.

        Returns:
            (all_ok: bool, broken_lines: list[int])   (1-based line numbers)
        """
        if not self.log_path.exists():
            return True, []

        broken: list[int] = []
        prev_hash = _GENESIS_HASH
        prev_line_bytes: bytes = b""

        with open(self.log_path, encoding="utf-8") as f:
            for lineno, raw_line in enumerate(f, 1):
                raw_line = raw_line.rstrip("\n")
                if not raw_line.strip():
                    continue
                try:
                    entry_dict = json.loads(raw_line)
                except json.JSONDecodeError:
                    _logger.error(f"[AuditLogger] Invalid JSON at line {lineno}")
                    broken.append(lineno)
                    continue

                stored_sig  = entry_dict.pop("hmac_sig", "")
                stored_prev = entry_dict.get("prev_hash", "")

                # Verify prev_hash link
                if lineno == 1:
                    if stored_prev != _GENESIS_HASH:
                        _logger.error(f"[AuditLogger] Chain break: line 1 prev_hash != genesis")
                        broken.append(lineno)
                else:
                    expected_prev = self._hash_bytes(prev_line_bytes)
                    if stored_prev != expected_prev:
                        _logger.error(
                            f"[AuditLogger] Chain break at line {lineno}: "
                            f"prev_hash mismatch"
                        )
                        broken.append(lineno)

                # Verify HMAC
                canonical = json.dumps(entry_dict, sort_keys=True, ensure_ascii=False)
                expected_sig = self._sign(canonical.encode())
                if stored_sig != expected_sig:
                    _logger.error(f"[AuditLogger] HMAC mismatch at line {lineno}")
                    broken.append(lineno)

                prev_line_bytes = raw_line.encode()

        all_ok = len(broken) == 0
        if all_ok:
            _logger.info("[AuditLogger] Chain verification PASSED")
        else:
            _logger.error(
                f"[AuditLogger] Chain verification FAILED | "
                f"broken_lines={broken}"
            )
        return all_ok, broken

    # -------------------------------------------------------------------------
    # Query helpers
    # -------------------------------------------------------------------------

    def read_recent(self, limit: int = 50) -> list[dict]:
        """Return up to `limit` most-recent audit entries (newest first)."""
        if not self.log_path.exists():
            return []
        lines: list[dict] = []
        with open(self.log_path, encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    lines.append(json.loads(raw_line))
                except json.JSONDecodeError:
                    pass
        return list(reversed(lines[-limit:]))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _sign(self, data: bytes) -> str:
        return hmac.new(self._key, data, hashlib.sha256).hexdigest()

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _read_last_hash(self) -> str:
        """Read the last line of the log and compute its hash, for chaining."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return _GENESIS_HASH
        last_line = b""
        with open(self.log_path, "rb") as f:
            for line in f:
                if line.strip():
                    last_line = line.rstrip(b"\r\n")
        if not last_line:
            return _GENESIS_HASH
        return self._hash_bytes(last_line)
