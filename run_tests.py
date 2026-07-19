"""
run_tests.py - quick sanity check against data/test_cases.json.

Not a substitute for real clinical validation (see README limitations
section) -- this only checks that the pipeline's tier output matches the
engineering expectation for each hand-written case, useful for catching
regressions while iterating.

Usage: python run_tests.py
"""

import json

from triage.gemma_client import GemmaClient
from triage.triage_engine import run_triage

def main():
    with open("data/test_cases.json") as f:
        cases = json.load(f)

    client = GemmaClient()
    print(f"Using backend: {client.backend}\n")

    passed = 0
    for case in cases:
        result = run_triage(case["input"], client=client)
        ok = result.tier.value == case["expected_tier"]
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}: expected={case['expected_tier']} got={result.tier.value}")
        if not ok:
            print(f"         input: {case['input']}")
            print(f"         rule_reason: {result.rule_reason}")
            print(f"         rationale: {result.rationale}")

    print(f"\n{passed}/{len(cases)} passed")


if __name__ == "__main__":
    main()
