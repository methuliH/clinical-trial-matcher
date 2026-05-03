"""
AI eligibility reasoning via the xAI Grok API (OpenAI-compatible interface).
"""
import json
import os
import re
from datetime import date

from openai import AsyncOpenAI

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.5-flash"

# Cap eligibility criteria fed to the model to avoid token blowout on huge trials.
_MAX_CRITERIA_CHARS = 6000

_SYSTEM_PROMPT = (
    "You are an expert oncology clinical trial matching specialist. "
    "You reason carefully through clinical trial eligibility criteria against a patient profile. "
    "Respond ONLY with a valid JSON object — no markdown fences, no prose outside the JSON."
)

_USER_TEMPLATE = """\
## Patient Profile
{patient_summary}

## Clinical Trial
NCT ID:  {nct_id}
Title:   {title}
Phase:   {phase}
Sponsor: {sponsor}

### Full Eligibility Criteria
{eligibility_criteria}

## Your Task
Assess whether this patient meets the above eligibility criteria.

Return a JSON object with EXACTLY these fields:
{{
  "match_score": <integer 0-100; 100 = perfect match, 0 = clearly ineligible>,
  "verdict": <one of: "eligible" | "likely_eligible" | "uncertain" | "likely_ineligible" | "ineligible">,
  "key_matches": [<brief phrase for each criterion the patient clearly meets>],
  "key_barriers": [<brief phrase for each criterion the patient fails or likely fails>],
  "unknown_criteria": [<brief phrase for each criterion that cannot be determined from available data>],
  "reasoning_summary": "<2-3 sentences of plain-English explanation>"
}}
"""

_VALID_VERDICTS = frozenset(
    {"eligible", "likely_eligible", "uncertain", "likely_ineligible", "ineligible"}
)


# ── Public API ────────────────────────────────────────────────────────────────

async def assess_eligibility(bundle: dict, trial: dict) -> dict:
    """
    Ask Grok to reason through a trial's eligibility criteria against a patient bundle.

    Returns a normalised dict with keys:
        match_score, verdict, key_matches, key_barriers,
        unknown_criteria, reasoning_summary
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _error_result("GEMINI_API_KEY environment variable is not set")

    criteria = trial.get("eligibility_criteria", "") or "(eligibility criteria not available)"
    if len(criteria) > _MAX_CRITERIA_CHARS:
        criteria = criteria[:_MAX_CRITERIA_CHARS] + "\n[...criteria truncated...]"

    prompt = _USER_TEMPLATE.format(
        patient_summary=_patient_summary(bundle),
        nct_id=trial.get("nct_id", ""),
        title=trial.get("title", ""),
        phase=", ".join(trial.get("phase", [])) or "N/A",
        sponsor=trial.get("sponsor", ""),
        eligibility_criteria=criteria,
    )

    try:
        async with AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=api_key) as client:
            resp = await client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        raw_text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return _error_result(str(exc))

    return _parse_response(raw_text)


# ── Patient summary builder ───────────────────────────────────────────────────

def _patient_summary(bundle: dict) -> str:
    patient = bundle["patient"]

    # Age
    birth_date = patient.get("birthDate", "")
    age_str = "unknown age"
    if birth_date:
        try:
            born = date.fromisoformat(birth_date)
            today = date.today()
            age = today.year - born.year - (
                (today.month, today.day) < (born.month, born.day)
            )
            age_str = f"{age}-year-old"
        except ValueError:
            pass
    gender = patient.get("gender", "unknown")

    lines = [f"{age_str} {gender}\n"]

    # Diagnoses
    if bundle["conditions"]:
        lines.append("DIAGNOSES:")
        for cond in bundle["conditions"]:
            code = cond.get("code", {})
            label = code.get("text") or _first_display(code.get("coding", []))
            clinical_status = (
                cond.get("clinicalStatus", {})
                    .get("coding", [{}])[0]
                    .get("code", "")
            )
            stage_label = ""
            for stage_entry in cond.get("stage", []):
                stage_label = _first_display(
                    stage_entry.get("summary", {}).get("coding", [])
                )
                if stage_label:
                    break
            parts = filter(None, [label, f"stage: {stage_label}" if stage_label else None])
            line = "  - " + " | ".join(parts)
            if clinical_status:
                line += f"  [{clinical_status}]"
            lines.append(line)

    # Biomarkers / clinical status
    if bundle["observations"]:
        lines.append("\nBIOMARKERS / CLINICAL STATUS:")
        for obs in bundle["observations"]:
            code = obs.get("code", {})
            label = code.get("text") or _first_display(code.get("coding", []))
            value = _obs_value(obs)
            if label:
                lines.append(f"  - {label}: {value}" if value else f"  - {label}: (no value)")

    # Prior / current treatments
    if bundle["medications"]:
        lines.append("\nPRIOR / CURRENT TREATMENTS:")
        for med in bundle["medications"]:
            med_cc = med.get("medicationCodeableConcept", {})
            name = med_cc.get("text") or _first_display(med_cc.get("coding", []))
            status = med.get("status", "")
            authored = med.get("authoredOn", "")
            detail = ", ".join(filter(None, [status, authored]))
            lines.append(f"  - {name} ({detail})" if detail else f"  - {name}")

    # Allergies
    if bundle["allergies"]:
        lines.append("\nALLERGIES:")
        for allergy in bundle["allergies"]:
            code = allergy.get("code", {})
            substance = code.get("text") or _first_display(code.get("coding", []))
            if substance:
                lines.append(f"  - {substance}")

    # Procedures
    if bundle["procedures"]:
        lines.append("\nPROCEDURES:")
        for proc in bundle["procedures"]:
            code = proc.get("code", {})
            name = code.get("text") or _first_display(code.get("coding", []))
            when = proc.get("performedDateTime", "")
            status = proc.get("status", "")
            detail = ", ".join(filter(None, [status, when]))
            lines.append(f"  - {name} ({detail})" if detail else f"  - {name}")

    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_display(coding_list: list[dict]) -> str:
    for coding in coding_list:
        if coding.get("display"):
            return coding["display"]
    return ""


def _obs_value(obs: dict) -> str:
    if "valueQuantity" in obs:
        vq = obs["valueQuantity"]
        return f"{vq.get('value', '')} {vq.get('unit', '')}".strip()
    if "valueCodeableConcept" in obs:
        vcc = obs["valueCodeableConcept"]
        return vcc.get("text") or _first_display(vcc.get("coding", []))
    if "valueString" in obs:
        return obs["valueString"]
    return ""


def _parse_response(raw: str) -> dict:
    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except json.JSONDecodeError:
                return _error_result(f"Could not parse model JSON: {raw[:200]}")
        else:
            return _error_result(f"No JSON found in response: {raw[:200]}")

    return _normalise(result)


def _normalise(raw: dict) -> dict:
    return {
        "match_score": max(0, min(100, int(raw.get("match_score") or 0))),
        "verdict": (
            raw.get("verdict") if raw.get("verdict") in _VALID_VERDICTS else "uncertain"
        ),
        "key_matches": [str(x) for x in raw.get("key_matches") or []],
        "key_barriers": [str(x) for x in raw.get("key_barriers") or []],
        "unknown_criteria": [str(x) for x in raw.get("unknown_criteria") or []],
        "reasoning_summary": str(raw.get("reasoning_summary") or ""),
    }


def _error_result(msg: str) -> dict:
    return {
        "match_score": 0,
        "verdict": "uncertain",
        "key_matches": [],
        "key_barriers": [],
        "unknown_criteria": [],
        "reasoning_summary": f"Reasoning unavailable: {msg}",
    }
