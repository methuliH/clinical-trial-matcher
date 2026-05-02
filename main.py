from fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP("clinical-trial-matcher")


@mcp.tool()
async def match_trials(
    fhir_base_url: str,
    fhir_access_token: str,
    max_results: int = 5
) -> dict:
    """
    Matches a patient to open clinical trials using their FHIR record.
    Returns ranked trials with AI-assessed eligibility scores.

    Args:
        fhir_base_url: Base URL of the FHIR server (from SHARP context)
        fhir_access_token: Bearer token for FHIR server auth (from SHARP context)
        max_results: Maximum number of trial matches to return (default 5)
    """
    # TODO Day 2: fetch FHIR patient bundle
    # TODO Day 3: query ClinicalTrials.gov
    # TODO Day 4: AI eligibility reasoning
    return {
        "status": "stub",
        "message": "match_trials tool registered successfully",
        "trials": []
    }


@mcp.tool()
async def explain_eligibility(
    nct_id: str,
    fhir_base_url: str,
    fhir_access_token: str
) -> dict:
    """
    Returns a criterion-by-criterion eligibility breakdown for a specific
    clinical trial against the patient's FHIR record.

    Args:
        nct_id: The ClinicalTrials.gov NCT identifier (e.g. NCT04567890)
        fhir_base_url: Base URL of the FHIR server (from SHARP context)
        fhir_access_token: Bearer token for FHIR server auth (from SHARP context)
    """
    # TODO Day 5: fetch trial criteria + patient bundle, run AI breakdown
    return {
        "status": "stub",
        "nct_id": nct_id,
        "criteria_breakdown": []
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
        "details": {}
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)