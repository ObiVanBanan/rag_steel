"""Compare the latest V4 evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.v4_constants import (
    DEFAULT_DEEPSEEK_RESULTS_PATH,
    DEFAULT_E2E_RESULTS_PATH,
    DEFAULT_RAG_RESULTS_PATH,
    DEFAULT_RESOLUTION_RESULTS_PATH,
    DEFAULT_SUMMARY_PATH,
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any] | None, key: str) -> float:
    if not payload:
        return 0.0
    return float(payload.get("summary", {}).get(key, 0.0))


def _validate_payloads(
    artifacts: dict[str, dict[str, Any] | None],
) -> None:
    missing = [name for name, payload in artifacts.items() if payload is None]
    if missing:
        raise ValueError(
            "Cannot compare V4 runs: missing artifacts for "
            + ", ".join(sorted(missing))
        )

    cases_by_name = {
        name: int(payload.get("summary", {}).get("cases", 0))
        for name, payload in artifacts.items()
        if payload is not None
    }
    sha_by_name = {
        name: str(payload.get("dataset_sha256", ""))
        for name, payload in artifacts.items()
        if payload is not None
    }

    if len(set(cases_by_name.values())) > 1:
        raise ValueError(
            "Cannot compare V4 runs: mismatched case counts "
            + json.dumps(cases_by_name, ensure_ascii=False)
        )
    if len(set(sha_by_name.values())) > 1:
        raise ValueError(
            "Cannot compare V4 runs: mismatched dataset sha256 "
            + json.dumps(sha_by_name, ensure_ascii=False)
        )


def compare_v4_results(
    *,
    deepseek_path: Path = DEFAULT_DEEPSEEK_RESULTS_PATH,
    resolution_path: Path = DEFAULT_RESOLUTION_RESULTS_PATH,
    rag_path: Path = DEFAULT_RAG_RESULTS_PATH,
    e2e_path: Path = DEFAULT_E2E_RESULTS_PATH,
    output_path: Path = DEFAULT_SUMMARY_PATH,
) -> str:
    deepseek = _load_json(deepseek_path)
    resolution = _load_json(resolution_path)
    rag = _load_json(rag_path)
    e2e = _load_json(e2e_path)
    _validate_payloads(
        {
            "deepseek": deepseek,
            "resolution": resolution,
            "rag": rag,
            "e2e": e2e,
        }
    )
    deepseek_cases = int(deepseek["summary"]["cases"])
    resolution_cases = int(resolution["summary"]["cases"])
    rag_cases = int(rag["summary"]["cases"])
    e2e_cases = int(e2e["summary"]["cases"])

    lines = [
        "# V4 Summary",
        "",
        "## Run Inputs",
        f"- DeepSeek: `{deepseek_path}`",
        f"- Resolution: `{resolution_path}`",
        f"- RAG: `{rag_path}`",
        f"- E2E: `{e2e_path}`",
        "",
        "## Key Metrics",
        f"- DeepSeek raw brand accuracy: `{_metric(deepseek, 'raw_brand_accuracy'):.4f}`",
        f"- DeepSeek article accuracy: `{_metric(deepseek, 'article_accuracy'):.4f}`",
        (
            f"- Resolution overall accuracy: "
            f"`{_metric(resolution, 'overall_resolution_accuracy'):.4f}`"
        ),
        (
            f"- Resolution false correction rate: "
            f"`{_metric(resolution, 'false_correction_rate'):.4f}`"
        ),
        f"- RAG status accuracy: `{_metric(rag, 'status_accuracy'):.4f}`",
        f"- RAG hard violation rate: `{_metric(rag, 'hard_violation_rate'):.4f}`",
        f"- E2E status accuracy: `{_metric(e2e, 'status_accuracy'):.4f}`",
        f"- E2E overall pass rate: `{_metric(e2e, 'overall_pass_rate'):.4f}`",
        "",
        "## Dataset",
        f"- DeepSeek cases: `{deepseek_cases}`",
        f"- Resolution cases: `{resolution_cases}`",
        f"- RAG cases: `{rag_cases}`",
        f"- E2E cases: `{e2e_cases}`",
    ]
    report = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare V4 evaluation outputs.")
    parser.add_argument("--deepseek", type=Path, default=DEFAULT_DEEPSEEK_RESULTS_PATH)
    parser.add_argument("--resolution", type=Path, default=DEFAULT_RESOLUTION_RESULTS_PATH)
    parser.add_argument("--rag", type=Path, default=DEFAULT_RAG_RESULTS_PATH)
    parser.add_argument("--e2e", type=Path, default=DEFAULT_E2E_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = compare_v4_results(
        deepseek_path=args.deepseek,
        resolution_path=args.resolution,
        rag_path=args.rag,
        e2e_path=args.e2e,
        output_path=args.output,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
