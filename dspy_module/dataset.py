"""
MSPDataset — Load and split evaluation dataset for DSPy training
------------------------------------------------------------------
Loads 80 queries from eval/datasets/*, excludes gold tasks,
and creates stratified train/dev split.

Split strategy:
  - Load all queries from 6 category files
  - Exclude gold task IDs (never train on these)
  - Stratified split by category: 70% train, 30% dev
  - Gold tasks remain separate for regression testing

Usage:
    dataset = MSPDataset()
    print(dataset.summary())  # total_train=42, total_dev=18, total_gold=20
    for example in dataset.train:
        print(example.query)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dspy
from loguru import logger


@dataclass
class DatasetSummary:
    """Summary statistics for the dataset split."""
    total_train: int
    total_dev: int
    total_gold: int
    total_queries: int
    split_by_category: dict[str, dict[str, int]]  # category -> {train, dev, gold}


class MSPDataset:
    """
    Loads enterprise RAG eval dataset and creates train/dev/gold splits.

    Attributes:
        train: list[dspy.Example] — 70% of non-gold queries
        dev: list[dspy.Example] — 30% of non-gold queries
        gold: list[dspy.Example] — reserved for regression testing (never train)
    """

    DATASETS_DIR = Path(__file__).parent.parent / "eval" / "datasets"
    CATEGORY_FILES = {
        "billing": "billing_queries.json",
        "contracts": "contracts_queries.json",
        "crm": "crm_queries.json",
        "psa": "psa_queries.json",
        "communications": "communications_queries.json",
        "cross_source": "cross_source_queries.json",
    }

    def __init__(self, seed: int = 42) -> None:
        """
        Load dataset and create splits.

        Args:
            seed: Random seed for reproducible split
        """
        random.seed(seed)

        # Load gold task IDs (never train on these)
        self.gold_ids = self._load_gold_ids()

        # Load all queries
        all_queries = self._load_all_queries()

        # Separate gold from trainable
        gold_queries = []
        trainable_queries = []

        for query_dict in all_queries:
            if query_dict["id"] in self.gold_ids:
                gold_queries.append(query_dict)
            else:
                trainable_queries.append(query_dict)

        # Stratified split by category
        self.train, self.dev = self._stratified_split(trainable_queries)

        # Convert gold queries to examples
        self.gold = [self._to_example(q) for q in gold_queries]

        logger.info(
            f"[Dataset] Loaded {len(all_queries)} total queries | "
            f"Train: {len(self.train)}, Dev: {len(self.dev)}, Gold: {len(self.gold)}"
        )

    def _load_gold_ids(self) -> set[str]:
        """Load gold task IDs from gold_tasks.json."""
        gold_file = self.DATASETS_DIR / "gold_tasks.json"

        if not gold_file.exists():
            logger.warning(f"Gold tasks file not found: {gold_file}")
            return set()

        try:
            with open(gold_file) as f:
                data = json.load(f)

            # Extract IDs from gold tasks
            if isinstance(data, list):
                return {item.get("id") for item in data if "id" in item}
            elif isinstance(data, dict) and "queries" in data:
                return {item.get("id") for item in data["queries"] if "id" in item}
            elif isinstance(data, dict) and "tasks" in data:
                return {item.get("id") for item in data["tasks"] if "id" in item}
            else:
                return set()
        except Exception as e:
            logger.error(f"Error loading gold tasks: {e}")
            return set()

    def _load_all_queries(self) -> list[dict]:
        """Load all queries from 6 category files."""
        all_queries = []

        for category, filename in self.CATEGORY_FILES.items():
            filepath = self.DATASETS_DIR / filename

            if not filepath.exists():
                logger.warning(f"Dataset file not found: {filepath}")
                continue

            try:
                with open(filepath) as f:
                    data = json.load(f)

                # Extract queries from standard format
                if isinstance(data, dict) and "queries" in data:
                    queries = data["queries"]
                else:
                    queries = data if isinstance(data, list) else []

                # Add category to each query
                for q in queries:
                    q["category"] = category
                    all_queries.append(q)

                logger.debug(f"[Dataset] Loaded {len(queries)} queries from {category}")

            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")

        return all_queries

    def _stratified_split(
        self, queries: list[dict], train_ratio: float = 0.7
    ) -> tuple[list[dspy.Example], list[dspy.Example]]:
        """
        Stratified split by category to ensure even distribution.

        Args:
            queries: All trainable queries (non-gold)
            train_ratio: Fraction to use for training (default 0.7)

        Returns:
            (train_examples, dev_examples)
        """
        # Group by category
        by_category = {}
        for q in queries:
            cat = q.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(q)

        train_examples = []
        dev_examples = []

        # Split each category
        for category, cat_queries in by_category.items():
            random.shuffle(cat_queries)
            split_idx = int(len(cat_queries) * train_ratio)
            train_cat = cat_queries[:split_idx]
            dev_cat = cat_queries[split_idx:]

            train_examples.extend([self._to_example(q) for q in train_cat])
            dev_examples.extend([self._to_example(q) for q in dev_cat])

            logger.debug(
                f"[Dataset] {category}: {len(train_cat)} train, {len(dev_cat)} dev"
            )

        return train_examples, dev_examples

    @staticmethod
    def _to_example(query_dict: dict) -> dspy.Example:
        """
        Convert raw query dict to dspy.Example with .with_inputs().

        Args:
            query_dict: Query from JSON file

        Returns:
            dspy.Example with input/output fields
        """
        example = dspy.Example(
            query=query_dict.get("query", ""),
            answer=query_dict.get("ground_truth", ""),
            expected_keywords=query_dict.get("expected_keywords", []),
            expected_source_types=query_dict.get("expected_source_types", []),
            category=query_dict.get("category", ""),
            difficulty=query_dict.get("difficulty", "medium"),
            query_id=query_dict.get("id", ""),
        )

        # Mark 'query' as the only input field
        return example.with_inputs("query")

    def summary(self) -> DatasetSummary:
        """Generate summary statistics."""
        split_by_cat = {}

        for example in self.train:
            cat = example.category
            if cat not in split_by_cat:
                split_by_cat[cat] = {"train": 0, "dev": 0, "gold": 0}
            split_by_cat[cat]["train"] += 1

        for example in self.dev:
            cat = example.category
            if cat not in split_by_cat:
                split_by_cat[cat] = {"train": 0, "dev": 0, "gold": 0}
            split_by_cat[cat]["dev"] += 1

        for example in self.gold:
            cat = example.category
            if cat not in split_by_cat:
                split_by_cat[cat] = {"train": 0, "dev": 0, "gold": 0}
            split_by_cat[cat]["gold"] += 1

        return DatasetSummary(
            total_train=len(self.train),
            total_dev=len(self.dev),
            total_gold=len(self.gold),
            total_queries=len(self.train) + len(self.dev) + len(self.gold),
            split_by_category=split_by_cat,
        )

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        s = self.summary()
        print("\n" + "=" * 70)
        print("DATASET SPLIT SUMMARY")
        print("=" * 70)
        print(f"Total queries: {s.total_queries}")
        print(f"  Train (70%): {s.total_train}")
        print(f"  Dev (30%):   {s.total_dev}")
        print(f"  Gold:        {s.total_gold} (never train)")
        print("\nBy category:")
        for cat in sorted(s.split_by_category.keys()):
            stats = s.split_by_category[cat]
            print(
                f"  {cat:18} — train: {stats['train']:2}, dev: {stats['dev']:2}, gold: {stats['gold']:2}"
            )
        print("=" * 70 + "\n")
