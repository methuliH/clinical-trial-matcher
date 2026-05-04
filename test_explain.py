"""
Integration test for the explain_eligibility tool.

Usage:
    python test_explain.py
"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv(encoding="utf-8-sig")

from main import explain_eligibility

NCT_ID     = "NCT05401110"
PATIENT_ID = "131999322"
FHIR_BASE  = "https://hapi.fhir.org/baseR4"

VERDICT_LABEL = {"met": "MET", "not_met": "NOT MET", "unknown": "UNKNOWN"}


async def run():
    print(f"explain_eligibility({NCT_ID}, patient={PATIENT_ID})\n")

    result = await explain_eligibility(
        nct_id=NCT_ID,
        fhir_base_url=FHIR_BASE,
        patient_id=PATIENT_ID,
    )

    if result["status"] == "error":
        print(f"ERROR: {result['message']}")
        return

    print(f"Trial  : {result['trial_title']}")
    print(f"Patient: {result['bundle_summary']['patient_name']}  "
          f"({result['bundle_summary']['gender']}, DOB {result['bundle_summary']['birth_date']})")
    print(f"Criteria evaluated: {result['total_criteria_evaluated']}\n")

    breakdown = result["criteria_breakdown"]
    if not breakdown:
        print("(no criteria returned)")
        return

    # Surface any error items before filtering
    for item in breakdown:
        if "error" in item:
            print(f"ERROR from reasoning layer: {item['error']}")
            return

    inc = [c for c in breakdown if c.get("type") == "inclusion"]
    exc = [c for c in breakdown if c.get("type") == "exclusion"]

    def print_section(title: str, items: list):
        if not items:
            return
        print(f"{'=' * 65}")
        print(f"  {title}")
        print(f"{'=' * 65}")
        for c in items:
            label = VERDICT_LABEL.get(c.get("verdict", "unknown"), "UNKNOWN")
            flag = {"MET": "[+]", "NOT MET": "[-]", "UNKNOWN": "[?]"}.get(label, "[?]")
            print(f"\n  {flag} [{label}]  #{c['n']}: {c['criterion']}")
            print(f"       Reason: {c['reason']}")
        print()

    print_section("INCLUSION CRITERIA", inc)
    print_section("EXCLUSION CRITERIA", exc)


if __name__ == "__main__":
    asyncio.run(run())
