"""
Invoice Revenue Forecaster
--------------------------
Uses Google TimeFM 2.5 (200M PyTorch) to project future monthly invoice
totals per client from 13 months of historical QuickBooks billing data.

TimeFM model weights (~800 MB) are downloaded from HuggingFace on the
first call to forecast().
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np


class InvoiceForecaster:
    """Forecast monthly revenue per client using TimeFM 2.5."""

    def __init__(self, invoice_path: str = "data/enterprise/invoices.json"):
        path = Path(invoice_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Invoice data not found at {invoice_path}. "
                "Run scripts/generate_enterprise_data.py first."
            )
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        self._records: list[dict] = data.get("records", data) if isinstance(data, dict) else data
        self._model = None  # lazy-loaded on first forecast() call

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_clients(self) -> list[dict]:
        """Return sorted deduplicated list of {client_id, client_name}."""
        seen: dict[str, str] = {}
        for rec in self._records:
            cid = rec.get("client_id", "")
            if cid and cid not in seen:
                seen[cid] = rec.get("client_name", cid)
        return [
            {"client_id": k, "client_name": v}
            for k, v in sorted(seen.items())
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _monthly_series(self, client_id: str) -> tuple[list[str], np.ndarray]:
        """
        Aggregate total_amount by YYYY-MM for the given client.
        Returns (sorted_labels, float32_array).
        """
        monthly: dict[str, float] = defaultdict(float)
        for rec in self._records:
            if rec.get("client_id") != client_id:
                continue
            date_str = rec.get("invoice_date", "")
            if len(date_str) < 7:
                continue
            month_key = date_str[:7]  # YYYY-MM
            # Sum line_items amounts
            for item in rec.get("line_items", []):
                monthly[month_key] += float(item.get("amount", 0))

        if not monthly:
            raise ValueError(f"No invoices found for client_id={client_id!r}")

        labels = sorted(monthly.keys())
        values = np.array([monthly[m] for m in labels], dtype=np.float32)
        return labels, values

    def _load_model(self):
        """Lazy-load TimeFM 2.5 200M PyTorch model."""
        import timesfm

        tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch"
        )
        cfg = timesfm.ForecastConfig(
            max_context=64,
            max_horizon=12,
            normalize_inputs=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            use_continuous_quantile_head=True,
        )
        tfm.compile(cfg)
        self._model = tfm

    @staticmethod
    def _next_months(last_label: str, n: int) -> list[str]:
        """Generate n YYYY-MM labels starting one month after last_label."""
        year, month = int(last_label[:4]), int(last_label[5:7])
        labels = []
        for _ in range(n):
            month += 1
            if month > 12:
                month = 1
                year += 1
            labels.append(f"{year:04d}-{month:02d}")
        return labels

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def forecast(self, client_id: str, horizon: int = 6) -> dict:
        """
        Run TimeFM forecast for one client.

        Returns a dict with keys:
            client_id, client_name, historical {dates, values},
            forecast {dates, values, lower_80, upper_80}
        """
        horizon = max(1, min(12, horizon))

        # Resolve client name
        client_name = client_id
        for c in self.get_clients():
            if c["client_id"] == client_id:
                client_name = c["client_name"]
                break

        hist_dates, hist_values = self._monthly_series(client_id)

        # Lazy-load model on first call
        if self._model is None:
            self._load_model()

        # Run TimeFM inference
        # forecast() returns (point_forecasts, quantile_forecasts)
        # point_forecasts: list of arrays, shape (horizon,) per input series
        # quantile_forecasts: list of arrays, shape (horizon, n_quantiles)
        point_preds, quantile_preds = self._model.forecast(
            horizon=horizon,
            inputs=[hist_values],
        )

        forecast_values = np.clip(np.array(point_preds[0], dtype=np.float32), 0, None)

        # quantile_preds[0] has shape (horizon, n_quantiles)
        # Default quantile levels: [0.1, 0.2, ..., 0.9]
        # index 0 => 10th pct (lower bound), index 8 => 90th pct (upper bound)
        q_array = np.array(quantile_preds[0], dtype=np.float32)  # (horizon, n_quantiles)
        lower_80 = np.clip(q_array[:, 0], 0, None)  # 10th percentile
        upper_80 = np.clip(q_array[:, 8], 0, None)  # 90th percentile

        forecast_dates = self._next_months(hist_dates[-1], horizon)

        return {
            "client_id": client_id,
            "client_name": client_name,
            "historical": {
                "dates": hist_dates,
                "values": [round(float(v), 2) for v in hist_values],
            },
            "forecast": {
                "dates": forecast_dates,
                "values": [round(float(v), 2) for v in forecast_values],
                "lower_80": [round(float(v), 2) for v in lower_80],
                "upper_80": [round(float(v), 2) for v in upper_80],
            },
        }
