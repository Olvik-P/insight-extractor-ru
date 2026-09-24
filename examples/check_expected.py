"""Sanity-check a real CallAnalysis result against hand-authored expectations.

This is NOT an accuracy benchmark. There is no labeled dataset for this
project yet, so there is nothing independent to compare the pipeline's
judgment against. What this script does check is narrower and honest about
its limits:

- The "expected" files were written by hand, before any real LLM run, based
  on facts deliberately built into each example transcript (a stated pain,
  a named objection, a concrete next-step date, etc.) plus a few
  deterministic values (``talk_ratio`` is computed by our own code, not the
  LLM, so its expected score is exact, not a guess).
- Passing these checks means the pipeline found the facts that are
  obviously present in the transcript and produced plausible scores. It
  does NOT mean the scores are "correct" in any calibrated sense — that
  needs real historical calls with independent human scoring (see
  design.md's open question about confidence-threshold tuning, and the
  LLM-judge calibration practice noted during exploration).

Usage:
    python examples/check_expected.py <result.json> <expected.json>
"""

from __future__ import annotations

import argparse
import json
import sys


def _contains_any(haystack: str, needles: list[str]) -> bool:
    haystack_lower = haystack.lower()
    return any(needle.lower() in haystack_lower for needle in needles)


def check(result: dict, expected: dict) -> list[str]:
    failures: list[str] = []
    summary = result.get("summary", {})

    if expected.get("client_pain_expected_none"):
        if summary.get("client_pain"):
            failures.append(
                f"expected no client_pain, got: {summary['client_pain']!r}"
            )
    elif "client_pain_keywords" in expected:
        pain = summary.get("client_pain") or ""
        keywords = expected["client_pain_keywords"]
        if not _contains_any(pain, keywords):
            failures.append(
                f"client_pain {pain!r} does not contain any of {keywords}"
            )

    if "objection_keyword_groups" in expected:
        objections_text = " | ".join(summary.get("objections", []))
        for group in expected["objection_keyword_groups"]:
            if not _contains_any(objections_text, group):
                failures.append(
                    f"objections {objections_text!r} missing expected group {group}"
                )

    if "min_action_items" in expected:
        count = len(summary.get("action_items", []))
        if count < expected["min_action_items"]:
            failures.append(
                f"expected >= {expected['min_action_items']} action_items, got {count}"
            )

    if "next_step_keywords" in expected:
        tasks_text = " | ".join(
            item.get("task", "") for item in summary.get("action_items", [])
        )
        keywords = expected["next_step_keywords"]
        if not _contains_any(tasks_text, keywords):
            failures.append(
                f"action item tasks {tasks_text!r} do not mention any of {keywords}"
            )

    criteria_by_id = {c["id"]: c for c in result.get("criteria", [])}
    for criterion_id, (lo, hi) in expected.get("criteria_score_ranges", {}).items():
        match = criteria_by_id.get(criterion_id)
        if match is None:
            failures.append(f"criterion {criterion_id!r} missing from result")
            continue
        if not (lo <= match["score"] <= hi):
            failures.append(
                f"{criterion_id} score {match['score']} outside expected range "
                f"[{lo}, {hi}]"
            )

    if "overall_score_range" in expected:
        lo, hi = expected["overall_score_range"]
        overall = result.get("overall_score")
        if not (lo <= overall <= hi):
            failures.append(
                f"overall_score {overall} outside expected range [{lo}, {hi}]"
            )

    return failures


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", help="Path to a CallAnalysis JSON result")
    parser.add_argument("expected_json", help="Path to the matching *.expected.json")
    args = parser.parse_args(argv)

    with open(args.result_json, encoding="utf-8") as f:
        result = json.load(f)
    with open(args.expected_json, encoding="utf-8") as f:
        expected = json.load(f)

    failures = check(result, expected)
    if failures:
        print(f"FAIL ({len(failures)} issue(s)):")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)

    print(
        "PASS — matches hand-authored expectations "
        "(a smoke check, not an accuracy benchmark)."
    )


if __name__ == "__main__":
    main()
