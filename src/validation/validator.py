"""
Document Validator
-------------------
Runs 7 quality checks on each RawDocument and returns typed
ValidatedDocument / RejectedDocument objects.

Checks (in order):
    1. Minimum content length
    2. Maximum content length
    3. Language detection
    4. Alphabetic character ratio
    5. Boilerplate detection
    6. Exact-duplicate (checksum)
    7. Composite quality score threshold
"""
from __future__ import annotations

from loguru import logger

from src.schemas import (
    CollectionStats,
    RawDocument,
    RejectedDocument,
    ValidatedDocument,
    ValidationResult,
)
from src.validation.quality_checks import (
    check_alphabetic_ratio,
    check_boilerplate,
    check_duplicate,
    check_language,
    check_max_length,
    check_min_length,
    check_url_format,
    compute_quality_score,
)


class DocumentValidator:
    """
    Stateful validator - tracks seen checksums across a batch to catch duplicates.

    Re-initialise between pipeline runs if you want a fresh dedup set.
    """

    def __init__(self, config: dict) -> None:
        self.min_length: int = config.get("min_content_length", 150)
        self.max_length: int = config.get("max_content_length", 500_000)
        self.allowed_languages: list[str] = config.get("allowed_languages", ["en"])
        self.quality_threshold: float = config.get("quality_score_threshold", 0.40)
        self._seen_checksums: set[str] = set()

    # --- Single Document ------------------------------------------------------

    def validate(self, doc: RawDocument) -> ValidationResult:
        """Run all checks on one document and return a ValidationResult."""
        passed: list[str] = []
        failed: list[str] = []
        notes: list[str] = []

        def _record(name: str, ok: bool, msg: str) -> None:
            (passed if ok else failed).append(name)
            notes.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")

        # 1. Min length
        ok, msg = check_min_length(doc.content, self.min_length)
        _record("min_length", ok, msg)

        # 2. Max length
        ok, msg = check_max_length(doc.content, self.max_length)
        _record("max_length", ok, msg)

        # 3. Language
        ok, msg = check_language(doc.content, self.allowed_languages)
        _record("language", ok, msg)

        # 4. Alphabetic ratio
        ok, _ratio, msg = check_alphabetic_ratio(doc.content)
        _record("alpha_ratio", ok, msg)

        # 5. Boilerplate
        ok, msg = check_boilerplate(doc.content)
        _record("boilerplate", ok, msg)

        # 6. Duplicate
        ok, msg = check_duplicate(doc.checksum, self._seen_checksums)
        _record("duplicate", ok, msg)

        # 7. URL format (soft check - only records, never blocks)
        ok, msg = check_url_format(doc.url)
        _record("url_format", ok, msg)

        # Composite quality score
        quality = compute_quality_score(doc.content, doc.title)
        if quality >= self.quality_threshold:
            _record("quality_score", True, f"Score {quality:.4f} >= threshold {self.quality_threshold}")
        else:
            _record("quality_score", False, f"Score {quality:.4f} < threshold {self.quality_threshold}")

        overall_pass = len(failed) == 0

        return ValidationResult(
            document_id=doc.id,
            passed=overall_pass,
            quality_score=quality,
            checks_passed=passed,
            checks_failed=failed,
            notes=notes,
        )

    # --- Batch ----------------------------------------------------------------

    def validate_batch(
        self, documents: list[RawDocument]
    ) -> tuple[list[ValidatedDocument], list[RejectedDocument]]:
        """
        Validate a list of RawDocuments.

        Returns (validated_docs, rejected_docs).
        """
        validated: list[ValidatedDocument] = []
        rejected: list[RejectedDocument] = []

        for doc in documents:
            result = self.validate(doc)
            if result.passed:
                validated.append(
                    ValidatedDocument.from_raw(doc, result.quality_score, result.notes)
                )
            else:
                rejected.append(
                    RejectedDocument.from_raw(doc, result.checks_failed)
                )

        logger.info(
            f"Validation: {len(validated)} passed / {len(rejected)} rejected "
            f"({len(documents)} total)"
        )
        return validated, rejected
