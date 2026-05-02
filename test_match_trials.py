"""
End-to-end integration test for match_trials.
Run from your local terminal (ClinicalTrials.gov blocks some cloud IPs):

    python test_match_trials.py
    python test_match_trials.py --patient-id 131999322 --max 5
"""
import argparse
import asyncio
import json

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

    print(f"\n=== Recruiting trials found: {result['total_trials_found']} ===")
    for i, trial in enumerate(result["trials"], 1):
        print(f"\n--- Trial {i} ---")
        print(f"  NCT ID  : {trial['nct_id']}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-id", default=DEFAULT_PATIENT_ID)
    parser.add_argument("--max", type=int, default=10, dest="max_results")
    args = parser.parse_args()
    asyncio.run(run(args.patient_id, args.max_results))
