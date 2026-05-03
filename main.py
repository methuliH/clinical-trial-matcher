import asyncio

from fastmcp import FastMCP
from dotenv import load_dotenv
import os

from fhir_client import FHIRClient, FHIRError
from trials_client import TrialsClient, TrialsError, extract_condition_terms
from reasoning import assess_eligibility

load_dotenv(encoding="utf-8-sig")  # utf-8-sig strips BOM if present

mcp = FastMCP("clinical-trial-matcher")


@mcp.tool()
async def match_trials(
    fhir_base_url: str,
    patient_id: str,
    fhir_access_token: str = "",
    max_results: int = 10,
) -> dict:
    """
    Matches a patient to open clinical trials using their FHIR record.
    Fetches the patient bundle from a FHIR R4 server, extracts conditions,
    queries ClinicalTrials.gov for recruiting trials, and returns the raw list.

    Args:
        fhir_base_url: Base URL of the FHIR server (e.g. https://hapi.fhir.org/baseR4)
        patient_id: FHIR Patient resource ID
        fhir_access_token: Bearer token for FHIR server auth (omit for public servers)
        max_results: Maximum number of trial matches to return (default 10)
    """
    # ── 1. Fetch FHIR patient bundle ──────────────────────────────────────────
    try:
        async with FHIRClient(fhir_base_url, fhir_access_token or None) as fhir:
            bundle = await fhir.get_patient_bundle(patient_id)
    except FHIRError as exc:
        return {"status": "error", "source": "fhir", "message": str(exc), "trials": []}

    # ── 2. Extract condition search terms from the bundle ─────────────────────
    condition_terms = extract_condition_terms(bundle["conditions"])

    # ── 3. Query ClinicalTrials.gov ───────────────────────────────────────────
    try:
        async with TrialsClient() as trials:
            trial_list = await trials.search(condition_terms, max_results=max_results)
    except TrialsError as exc:
        return {
            "status": "error",
            "source": "clinicaltrials",
            "message": str(exc),
            "condition_search_terms": condition_terms,
            "bundle_summary": _bundle_summary(bundle),
            "trials": [],
        }

    # ── 4. AI eligibility reasoning (all trials in parallel) ─────────────────
    scores = await asyncio.gather(
        *[assess_eligibility(bundle, t) for t in trial_list],
        return_exceptions=True,
    )
    for trial, score in zip(trial_list, scores):
        if isinstance(score, Exception):
            trial["eligibility"] = {
                "match_score": 0, "verdict": "uncertain",
                "key_matches": [], "key_barriers": [], "unknown_criteria": [],
                "reasoning_summary": f"Reasoning failed: {score}",
            }
        else:
            trial["eligibility"] = score

    trial_list.sort(
        key=lambda t: t["eligibility"].get("match_score", 0), reverse=True
    )

    return {
        "status": "ok",
        "patient_id": patient_id,
        "condition_search_terms": condition_terms,
        "bundle_summary": _bundle_summary(bundle),
        "total_trials_found": len(trial_list),
        "trials": trial_list,
    }


@mcp.tool()
async def explain_eligibility(
    nct_id: str,
    fhir_base_url: str,
    patient_id: str,
    fhir_access_token: str = "",
) -> dict:
    """
    Returns a criterion-by-criterion eligibility breakdown for a specific
    clinical trial against the patient's FHIR record.

    Args:
        nct_id: The ClinicalTrials.gov NCT identifier (e.g. NCT04567890)
        fhir_base_url: Base URL of the FHIR server
        patient_id: FHIR Patient resource ID
        fhir_access_token: Bearer token for FHIR server auth (omit for public servers)
    """
    try:
        async with FHIRClient(fhir_base_url, fhir_access_token or None) as fhir:
            bundle = await fhir.get_patient_bundle(patient_id)
    except FHIRError as exc:
        return {"status": "error", "message": str(exc), "nct_id": nct_id, "criteria_breakdown": []}

    # TODO Day 5: fetch trial criteria from ClinicalTrials.gov + run AI breakdown
    return {
        "status": "ok",
        "nct_id": nct_id,
        "patient_id": patient_id,
        "bundle_summary": _bundle_summary(bundle),
        "criteria_breakdown": [],
    }


@mcp.tool()
async def get_trial_details(nct_id: str) -> dict:
    """
    Returns full details for a clinical trial by NCT ID — summary,
    eligibility criteria, phase, sites, and contact information.

    Args:
        nct_id: The ClinicalTrials.gov NCT identifier (e.g. NCT04567890)
    """
    # TODO Day 3: fetch from ClinicalTrials.gov v2 API
    return {
        "status": "stub",
        "nct_id": nct_id,
        "details": {},
    }


def _bundle_summary(bundle: dict) -> dict:
    patient = bundle["patient"]
    name_parts = []
    for n in patient.get("name", []):
        family = n.get("family", "")
        given = " ".join(n.get("given", []))
        name_parts.append(f"{given} {family}".strip())

    return {
        "patient_name": name_parts[0] if name_parts else "unknown",
        "birth_date": patient.get("birthDate", ""),
        "gender": patient.get("gender", ""),
        "conditions": len(bundle["conditions"]),
        "observations": len(bundle["observations"]),
        "medications": len(bundle["medications"]),
        "allergies": len(bundle["allergies"]),
        "procedures": len(bundle["procedures"]),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
