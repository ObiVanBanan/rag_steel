"""Compare the latest V3 DeepSeek, RAG, and E2E reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.v3_constants import DEFAULT_RESULTS_DIR, DEFAULT_SUMMARY_PATH


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_json(prefix: str) -> Path | None:
    candidates = sorted(
        DEFAULT_RESULTS_DIR.glob(f"{prefix}*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _metric(payload: dict[str, Any], key: str) -> float:
    return float(payload.get("summary", {}).get(key, 0.0))


def compare_v3_results(
    *,
    deepseek_path: Path | None = None,
    rag_path: Path | None = None,
    e2e_path: Path | None = None,
    output_md: Path = DEFAULT_SUMMARY_PATH,
) -> dict[str, Any]:
    deepseek_path = deepseek_path or _latest_json("deepseek_v3_")
    rag_path = rag_path or _latest_json("rag_v3_")
    e2e_path = e2e_path or _latest_json("e2e_v3_")
    if deepseek_path is None or rag_path is None or e2e_path is None:
        raise FileNotFoundError("Missing one or more V3 result JSON files")

    deepseek = _load_json(deepseek_path)
    rag = _load_json(rag_path)
    e2e = _load_json(e2e_path)

    lines = [
        "# V3 Summary",
        "",
        f"Generated at `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Core",
        f"- DeepSeek hard exact: `{_metric(deepseek, 'hard_exact_match_rate'):.4f}`",
        f"- DeepSeek brand accuracy: `{_metric(deepseek, 'brand_accuracy'):.4f}`",
        f"- RAG preferred hit@1: `{_metric(rag, 'preferred_hit@1'):.4f}`",
        f"- RAG preferred hit@5: `{_metric(rag, 'preferred_hit@5'):.4f}`",
        f"- RAG MRR: `{_metric(rag, 'MRR'):.4f}`",
        f"- E2E preferred hit@1: `{_metric(e2e, 'e2e_preferred_hit@1'):.4f}`",
        f"- E2E preferred hit@5: `{_metric(e2e, 'e2e_preferred_hit@5'):.4f}`",
        f"- E2E overall pass: `{_metric(e2e, 'overall_pass_rate'):.4f}`",
        f"- E2E strict overall pass: `{_metric(e2e, 'strict_overall_pass_rate'):.4f}`",
        "",
        "## Dataset",
        f"- DeepSeek cases: `{deepseek.get('summary', {}).get('cases', 0)}`",
        f"- RAG cases: `{rag.get('summary', {}).get('cases', 0)}`",
        f"- E2E cases: `{e2e.get('summary', {}).get('cases', 0)}`",
        "",
        "## Latency",
        f"- DeepSeek p95: `{_metric(deepseek, 'latency_p95_ms'):.1f}` ms",
        f"- RAG wall clock p95: `{_metric(rag, 'wall_clock_p95_ms'):.1f}` ms",
        f"- E2E wall clock p95: `{_metric(e2e, 'wall_clock_p95_ms'):.1f}` ms",
        "",
        "## Failure Stages",
    ]
    for label, payload in (("DeepSeek", deepseek), ("RAG", rag), ("E2E", e2e)):
        lines.append(f"- {label}: {payload.get('failure_counts', {})}")

    report = "\n".join(lines).rstrip() + "\n"
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")
    return {
        "deepseek_path": str(deepseek_path),
        "rag_path": str(rag_path),
        "e2e_path": str(e2e_path),
        "output_md": str(output_md),
        "report": report,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare V3 evaluation results.")
    parser.add_argument("--deepseek", type=Path, default=None)
    parser.add_argument("--rag", type=Path, default=None)
    parser.add_argument("--e2e", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    payload = compare_v3_results(
        deepseek_path=args.deepseek,
        rag_path=args.rag,
        e2e_path=args.e2e,
        output_md=args.output_md,
    )
    print(payload["report"])
    print(payload["output_md"])
    return 0


__all__ = ["compare_v3_results"]


if __name__ == "__main__":
    raise SystemExit(main())
