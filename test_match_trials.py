"""
End-to-end integration test for match_trials.
Run from your local terminal (ClinicalTrials.gov blocks some cloud IPs):

    python test_match_trials.py
    python test_match_trials.py --patient-id 131999322 --max 5
"""
import argparse
import asyncio
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from main import match_trials

DEFAULT_FHIR_URL = "https://hapi.fhir.org/baseR4"
DEFAULT_PATIENT_ID = "131999322"


async def run(patient_id: str, max_results: int):
    print(f"Fetching FHIR bundle for patient {patient_id} ...")
    print(f"Then querying ClinicalTrials.gov (max {max_results} results) ...\n")

    result = await match_trials(
        fhir_base_url=DEFAULT_FHIR_URL,
        patient_id=patient_id,
        max_results=max_results,
    )

    if result["status"] == "error":
        print(f"ERROR [{result.get('source', '?')}]: {result['message']}")
        if result.get("bundle_summary"):
            print("\nFHIR bundle was fetched successfully:")
            print(json.dumps(result["bundle_summary"], indent=2))
        return

    print("=== Patient summary ===")
    print(json.dumps(result["bundle_summary"], indent=2))

    print(f"\n=== Condition search terms sent to ClinicalTrials.gov ===")
    print(json.dumps(result["condition_search_terms"], indent=2))

    HIGHLIGHT = {"NCT05401110", "NCT06864624"}

    print(f"\n=== Recruiting trials found: {result['total_trials_found']} ===")
    for i, trial in enumerate(result["trials"], 1):
        nct = trial["nct_id"]
        marker = " ★ REQUESTED" if nct in HIGHLIGHT else ""
        print(f"\n{'='*60}")
        print(f"--- Trial {i}{marker} ---")
        print(f"  NCT ID  : {nct}")
        print(f"  Title   : {trial['title']}")
        print(f"  Phase   : {trial['phase']}")
        print(f"  Sponsor : {trial['sponsor']}")
        print(f"  Summary : {trial['brief_summary'][:200]}...")
        print(f"  Sites   : {len(trial['locations'])} location(s)")
        if trial["locations"]:
            for loc in trial["locations"][:3]:
                print(f"            - {loc}")
        crit = trial["eligibility_criteria"]
        print(f"  Criteria: {crit[:300]}{'...' if len(crit) > 300 else ''}")

        elig = trial.get("eligibility", {})
        print(f"\n  --- Eligibility Reasoning ---")
        print(f"  Match Score : {elig.get('match_score', 'N/A')} / 100")
        print(f"  Verdict     : {elig.get('verdict', 'N/A')}")
        matches = elig.get("key_matches") or []
        barriers = elig.get("key_barriers") or []
        unknowns = elig.get("unknown_criteria") or []
        if matches:
            print(f"  Key Matches :")
            for m in matches:
                print(f"    [+] {m}")
        if barriers:
            print(f"  Key Barriers:")
            for b in barriers:
                print(f"    [-] {b}")
        if unknowns:
            print(f"  Unknown     :")
            for u in unknowns:
                print(f"    ? {u}")
        print(f"  Summary     : {elig.get('reasoning_summary', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default=DEFAULT_PATIENT_ID)
    parser.add_argument("--max", type=int, default=10, dest="max_results")
    args = parser.parse_args()
    asyncio.run(run(args.patient_id, args.max_results))
