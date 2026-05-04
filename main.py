import asyncio

from fastmcp import FastMCP
from dotenv import load_dotenv
import os

from fhir_client import FHIRClient, FHIRError
from trials_client import TrialsClient, TrialsError, extract_condition_terms
from reasoning import assess_eligibility, breakdown_eligibility

load_dotenv(encoding="utf-8-sig")  # utf-8-sig strips BOM if present

mcp = FastMCP("clinical-trial-matcher")

# ── SHARP/FHIR context capability ────────────────────────────────────────────
# Advertise the ai.promptopinion/fhir-context extension so SMART-on-FHIR
# clients (e.g. Prompt Opinion) know which FHIR scopes this server needs and
# can inject FHIR context via HTTP headers instead of requiring explicit params.

_FHIR_SCOPES = [
    "patient/Patient.rs",
    "patient/Condition.rs",
    "patient/Observation.rs",
    "patient/MedicationRequest.rs",
    "patient/AllergyIntolerance.rs",
    "patient/Procedure.rs",
]

_orig_get_caps = mcp._mcp_server.get_capabilities


def _get_caps_with_fhir(notification_options, experimental_capabilities):
    caps = _orig_get_caps(notification_options, experimental_capabilities)
    existing = caps.experimental or {}
    caps.experimental = {
        **existing,
        "ai.promptopinion/fhir-context": {"requiredScopes": _FHIR_SCOPES},
    }
    return caps


mcp._mcp_server.get_capabilities = _get_caps_with_fhir


# ── Header helper ─────────────────────────────────────────────────────────────

def _fhir_headers() -> dict:
    """Return FHIR context from HTTP headers; empty strings when unavailable."""
    try:
        from fastmcp.server.dependencies import get_http_request
        req = get_http_request()
        return {
            "fhir_base_url": req.headers.get("x-fhir-server-url", ""),
            "patient_id": req.headers.get("x-patient-id", ""),
            "fhir_access_token": req.headers.get("x-fhir-access-token", ""),
        }
    except RuntimeError:
        return {"fhir_base_url": "", "patient_id": "", "fhir_access_token": ""}


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def match_trials(
    fhir_base_url: str = "",
    patient_id: str = "",
    fhir_access_token: str = "",
    max_results: int = 10,
) -> dict:
    """
    Matches a patient to open clinical trials using their FHIR record.
    Fetches the patient bundle from a FHIR R4 server, extracts conditions,
    queries ClinicalTrials.gov for recruiting trials, and returns the raw list.

    FHIR context can be supplied either as explicit parameters or via HTTP headers
    (X-FHIR-Server-URL, X-Patient-ID, X-FHIR-Access-Token). Headers take
    precedence when both are provided.

    Args:
        fhir_base_url: Base URL of the FHIR server (e.g. https://hapi.fhir.org/baseR4)
        patient_id: FHIR Patient resource ID
        fhir_access_token: Bearer token for FHIR server auth (omit for public servers)
        max_results: Maximum number of trial matches to return (default 10)
    """
    h = _fhir_headers()
    fhir_base_url = h["fhir_base_url"] or fhir_base_url
    patient_id = h["patient_id"] or patient_id
    fhir_access_token = h["fhir_access_token"] or fhir_access_token

    if not fhir_base_url or not patient_id:
        return {
            "status": "error",
            "source": "params",
            "message": "fhir_base_url and patient_id are required (via params or headers)",
            "trials": [],
        }

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
    fhir_base_url: str = "",
    patient_id: str = "",
    fhir_access_token: str = "",
) -> dict:
    """
    Returns a criterion-by-criterion eligibility breakdown for a specific
    clinical trial against the patient's FHIR record.

    FHIR context can be supplied either as explicit parameters or via HTTP headers
    (X-FHIR-Server-URL, X-Patient-ID, X-FHIR-Access-Token). Headers take
    precedence when both are provided.

    Args:
        nct_id: The ClinicalTrials.gov NCT identifier (e.g. NCT04567890)
        fhir_base_url: Base URL of the FHIR server
        patient_id: FHIR Patient resource ID
        fhir_access_token: Bearer token for FHIR server auth (omit for public servers)
    """
    h = _fhir_headers()
    fhir_base_url = h["fhir_base_url"] or fhir_base_url
    patient_id = h["patient_id"] or patient_id
    fhir_access_token = h["fhir_access_token"] or fhir_access_token

    if not fhir_base_url or not patient_id:
        return {
            "status": "error",
            "message": "fhir_base_url and patient_id are required (via params or headers)",
            "nct_id": nct_id,
            "criteria_breakdown": [],
        }

    try:
        async with FHIRClient(fhir_base_url, fhir_access_token or None) as fhir:
            bundle = await fhir.get_patient_bundle(patient_id)
    except FHIRError as exc:
        return {"status": "error", "message": str(exc), "nct_id": nct_id, "criteria_breakdown": []}

    try:
        async with TrialsClient() as trials:
            trial = await trials.get_by_nct_id(nct_id)
    except TrialsError as exc:
        return {"status": "error", "message": str(exc), "nct_id": nct_id, "criteria_breakdown": []}

    if trial is None:
        return {"status": "error", "message": f"{nct_id} not found on ClinicalTrials.gov",
                "nct_id": nct_id, "criteria_breakdown": []}

    criteria_breakdown = await breakdown_eligibility(bundle, trial)

    return {
        "status": "ok",
        "nct_id": nct_id,
        "patient_id": patient_id,
        "trial_title": trial["title"],
        "bundle_summary": _bundle_summary(bundle),
        "total_criteria_evaluated": len(criteria_breakdown),
        "criteria_breakdown": criteria_breakdown,
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
