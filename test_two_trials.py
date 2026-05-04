"""
Focused integration test: fetch FHIR bundle for one patient, then run
AI eligibility reasoning against two specific NCT IDs.

Usage:
    python test_two_trials.py
"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")

from fhir_client import FHIRClient
from trials_client import TrialsClient
from reasoning import assess_eligibility

FHIR_BASE = "https://hapi.fhir.org/baseR4"
PATIENT_ID = "131999322"
TARGET_NCTS = ["NCT05401110", "NCT06864624"]


async def run():
    print(f"[1/3] Fetching FHIR bundle for patient {PATIENT_ID} ...")
    async with FHIRClient(FHIR_BASE) as fhir:
        bundle = await fhir.get_patient_bundle(PATIENT_ID)

    patient = bundle["patient"]
    print(f"      Patient: {patient.get('name', [{}])[0].get('family', '?')}, "
          f"DOB {patient.get('birthDate', '?')}, gender {patient.get('gender', '?')}")
    print(f"      Conditions: {len(bundle['conditions'])}  |  "
          f"Observations: {len(bundle['observations'])}  |  "
          f"Medications: {len(bundle['medications'])}\n")

    print(f"[2/3] Fetching {len(TARGET_NCTS)} trials from ClinicalTrials.gov ...")
    async with TrialsClient() as trials:
        fetched = []
        for nct_id in TARGET_NCTS:
            trial = await trials.get_by_nct_id(nct_id)
            if trial:
                fetched.append(trial)
                print(f"      {nct_id}  -> {trial['title'][:70]}...")
            else:
                print(f"      {nct_id}  -> NOT FOUND")

    if not fetched:
        print("No trials fetched; aborting.")
        return

    print(f"\n[3/3] Running AI eligibility reasoning on {len(fetched)} trial(s) via Groq ...\n")

    for i, trial in enumerate(fetched, 1):
        print(f"Running reasoning for trial {i}/{len(fetched)}: {trial['nct_id']} ...")
        elig = await assess_eligibility(bundle, trial)

        print()
        print("=" * 65)
        print(f"  NCT ID      : {trial['nct_id']}")
        print(f"  Title       : {trial['title']}")
        print(f"  Phase       : {trial['phase']}")
        print(f"  Sponsor     : {trial['sponsor']}")
        print(f"  Sites       : {len(trial['locations'])} location(s)")
        for loc in trial["locations"][:3]:
            print(f"                - {loc}")
        crit = trial["eligibility_criteria"]
        print(f"  Criteria    : {crit[:300]}{'...' if len(crit) > 300 else ''}")
        print()
        print(f"  --- Eligibility Reasoning ---")
        print(f"  Match Score : {elig['match_score']} / 100")
        print(f"  Verdict     : {elig['verdict']}")
        for m in elig.get("key_matches") or []:
            print(f"    [+] {m}")
        for b in elig.get("key_barriers") or []:
            print(f"    [-] {b}")
        for u in elig.get("unknown_criteria") or []:
            print(f"    [?] {u}")
        print(f"  Summary     : {elig['reasoning_summary']}")
        print("=" * 65)
        print()


if __name__ == "__main__":
    asyncio.run(run())
