"""
Collection Pipeline - Phase I
------------------------------
Orchestrates all configured collectors, aggregates RawDocuments,
and persists them to disk for the validation stage.

Collectors registered:
  Research  : ArXiv, Wikipedia, RSS
  Enterprise: Billing (QuickBooks), PSA (ConnectWise), CRM (HubSpot),
              Communications (Exchange), Contracts (SharePoint)
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

from src.collection.arxiv_collector import ArXivCollector
from src.collection.billing_collector import BillingCollector
from src.collection.comms_collector import CommsCollector
from src.collection.contracts_collector import ContractsCollector
from src.collection.crm_collector import CRMCollector
from src.collection.psa_collector import PSACollector
from src.collection.rss_collector import RSSCollector
from src.collection.wikipedia_collector import WikipediaCollector
from src.schemas import CollectionStats, RawDocument
from src.utils.helpers import ensure_dirs, save_json

console = Console()


class CollectionPipeline:
    """
    Orchestrates all data collectors defined in config.

    Flow:
        1. health_check()   - verify sources are reachable
        2. collect_all()    - run all collectors, yield RawDocuments
        3. _save_raw()      - persist to data/raw/ + write manifest
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.raw_dir = Path(config.get("storage", {}).get("raw_dir", "data/raw"))
        self.stats = CollectionStats()
        self.documents: list[RawDocument] = []

        self.collectors: list = []
        col_cfg = config.get("collection", {})

        # --- Research sources ------------------------------------------------
        if col_cfg.get("arxiv", {}).get("enabled", False):
            self.collectors.append(ArXivCollector(col_cfg["arxiv"]))
            logger.debug("ArXivCollector enabled")

        if col_cfg.get("wikipedia", {}).get("enabled", False):
            self.collectors.append(WikipediaCollector(col_cfg["wikipedia"]))
            logger.debug("WikipediaCollector enabled")

        if col_cfg.get("rss", {}).get("enabled", False):
            self.collectors.append(RSSCollector(col_cfg["rss"]))
            logger.debug("RSSCollector enabled")

        # --- Enterprise sources ----------------------------------------------
        if col_cfg.get("billing", {}).get("enabled", False):
            self.collectors.append(BillingCollector(col_cfg["billing"]))
            logger.debug("BillingCollector enabled")

        if col_cfg.get("psa", {}).get("enabled", False):
            self.collectors.append(PSACollector(col_cfg["psa"]))
            logger.debug("PSACollector enabled")

        if col_cfg.get("crm", {}).get("enabled", False):
            self.collectors.append(CRMCollector(col_cfg["crm"]))
            logger.debug("CRMCollector enabled")

        if col_cfg.get("communications", {}).get("enabled", False):
            self.collectors.append(CommsCollector(col_cfg["communications"]))
            logger.debug("CommsCollector enabled")

        if col_cfg.get("contracts", {}).get("enabled", False):
            self.collectors.append(ContractsCollector(col_cfg["contracts"]))
            logger.debug("ContractsCollector enabled")

    # --- Public API -----------------------------------------------------------

    async def run_health_checks(self) -> dict[str, bool]:
        """Probe each data source and report reachability."""
        logger.info("Running source health checks...")
        results: dict[str, bool] = {}

        for collector in self.collectors:
            name = collector.__class__.__name__
            ok = await collector.health_check()
            results[name] = ok
            icon = "[OK]" if ok else "[FAIL]"
            level = "INFO" if ok else "WARNING"
            logger.log(level, f"  {icon} {name}: {'healthy' if ok else 'UNREACHABLE'}")

        return results

    async def collect_all(self) -> list[RawDocument]:
        """
        Run every enabled collector sequentially, aggregate all documents,
        de-duplicate by checksum, then persist to disk.
        """
        ensure_dirs(self.raw_dir)
        self.stats.started_at = datetime.utcnow()
        logger.info(f"Starting collection with {len(self.collectors)} collector(s)...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            for collector in self.collectors:
                name = collector.__class__.__name__
                task_id = progress.add_task(f"[cyan]{name}[/cyan]", total=None)
                batch: list[RawDocument] = []

                async for doc in collector.safe_collect():
                    batch.append(doc)
                    progress.update(
                        task_id,
                        description=f"[cyan]{name}[/cyan] - {len(batch)} docs",
                    )

                key = collector.source_type.value
                self.stats.by_source[key] = (
                    self.stats.by_source.get(key, 0) + len(batch)
                )
                self.documents.extend(batch)
                progress.update(task_id, completed=True)
                logger.info(f"[{name}] Collected {len(batch)} document(s)")

        # Global deduplication by checksum
        before = len(self.documents)
        seen: set[str] = set()
        unique: list[RawDocument] = []
        for doc in self.documents:
            if doc.checksum not in seen:
                seen.add(doc.checksum)
                unique.append(doc)
        self.documents = unique
        dupes = before - len(unique)
        if dupes:
            logger.info(f"Removed {dupes} exact duplicate(s)")

        self.stats.total_collected = len(self.documents)
        self.stats.completed_at = datetime.utcnow()

        await self._save_raw()
        return self.documents

    # --- Persistence ----------------------------------------------------------

    async def _save_raw(self) -> None:
        """Persist all raw documents as individual JSON files + a manifest."""
        logger.info(f"Saving {len(self.documents)} raw documents -> {self.raw_dir}/")

        for doc in self.documents:
            safe_id = doc.id.replace("/", "_").replace(":", "_").replace(" ", "_")
            path = self.raw_dir / f"{doc.source_type.value}_{safe_id}.json"
            save_json(doc.model_dump(mode="json"), path)

        manifest = {
            "run_id": self.stats.run_id,
            "collected_at": self.stats.started_at.isoformat(),
            "completed_at": self.stats.completed_at.isoformat() if self.stats.completed_at else None,
            "total_documents": len(self.documents),
            "by_source": self.stats.by_source,
            "document_ids": [d.id for d in self.documents],
        }
        save_json(manifest, self.raw_dir / "manifest.json")
        logger.info(f"Manifest written -> {self.raw_dir}/manifest.json")
