"""
AI eligibility reasoning via the Groq API (OpenAI-compatible interface).
"""
import asyncio
import json
import os
import re
from datetime import date

from openai import AsyncOpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Cap eligibility criteria fed to the model to avoid token blowout on huge trials.
_MAX_CRITERIA_CHARS = 6000
# Max individual criteria sent in a single breakdown call.
_MAX_CRITERIA_COUNT = 25

# Serialise calls to stay comfortably within Groq's free-tier RPM limit.
_call_lock = asyncio.Lock()

_VALID_BREAKDOWN_VERDICTS = frozenset({"met", "not_met", "unknown"})

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

_BREAKDOWN_SYSTEM_PROMPT = (
    "You are an expert oncology clinical trial eligibility specialist. "
    "Evaluate each numbered criterion individually against the patient profile provided. "
    "Respond ONLY with a valid JSON object — no markdown fences, no prose outside the JSON."
)

_BREAKDOWN_USER_TEMPLATE = """\
## Patient Profile
{patient_summary}

## Task
Evaluate every numbered criterion below against the patient profile.

Return a JSON object with a single key "breakdown" whose value is an array. \
Each array element must have EXACTLY these fields:
{{
  "n":       <criterion number as integer>,
  "type":    <"inclusion" or "exclusion">,
  "verdict": <"met" | "not_met" | "unknown">,
  "reason":  "<one concise sentence grounded in the patient data>"
}}

Use "unknown" only when the patient record genuinely lacks the data needed to decide.

## Eligibility Criteria
{criteria_list}
"""


# ── Public API ────────────────────────────────────────────────────────────────

async def assess_eligibility(bundle: dict, trial: dict) -> dict:
    """
    Ask Groq to reason through a trial's eligibility criteria against a patient bundle.

    Returns a normalised dict with keys:
        match_score, verdict, key_matches, key_barriers,
        unknown_criteria, reasoning_summary
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _error_result("GROQ_API_KEY environment variable is not set")

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
        async with _call_lock:
            async with AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=api_key, timeout=120.0) as client:
                resp = await client.chat.completions.create(
                    model=GROQ_MODEL,
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


async def breakdown_eligibility(bundle: dict, trial: dict) -> list[dict]:
    """
    Return a per-criterion eligibility breakdown for one trial against a patient bundle.

    Each item: {n, type, criterion, verdict, reason}
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return [{"error": "GROQ_API_KEY not set"}]

    raw_criteria = _split_criteria(trial.get("eligibility_criteria", ""))
    if not raw_criteria:
        return []

    criteria_to_eval = raw_criteria[:_MAX_CRITERIA_COUNT]
    criteria_list = "\n".join(
        f"{i + 1}. [{c['type'].upper()}] {c['text']}"
        for i, c in enumerate(criteria_to_eval)
    )

    prompt = _BREAKDOWN_USER_TEMPLATE.format(
        patient_summary=_patient_summary(bundle),
        criteria_list=criteria_list,
    )

    try:
        async with _call_lock:
            async with AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=api_key, timeout=120.0) as client:
                resp = await client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": _BREAKDOWN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
        raw_text = resp.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        return [{"error": str(exc)}]

    return _parse_breakdown(raw_text, criteria_to_eval)


# ── Patient summary builder ───────────────────────────────────────────────────

def _patient_summary(bundle: dict) -> str:
    patient = bundle["patient"]

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

    lines = [f"PATIENT: {age_str} {gender}"]

    # ── Molecular biomarkers first — these are the most eligibility-critical ──
    mol_obs = [
        o for o in bundle["observations"]
        if any(
            c.get("code") in ("69548-6", "81311-2", "85319-2", "51194-4", "85318-4")
            for c in o.get("code", {}).get("coding", [])
        )
    ]
    perf_obs = [o for o in bundle["observations"] if o not in mol_obs]

    if mol_obs:
        lines.append("\nMOLECULAR BIOMARKERS (*** READ CAREFULLY — CRITICAL FOR ELIGIBILITY ***):")
        for obs in mol_obs:
            label = obs["_label"]
            value = obs["_value"]
            lines.append(f"  *** {label}: {value if value else '(no value recorded)'} ***")

    # ── Diagnoses ──────────────────────────────────────────────────────────────
    if bundle["conditions"]:
        lines.append("\nDIAGNOSES:")
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

    # ── Performance / other clinical observations ──────────────────────────────
    if perf_obs:
        lines.append("\nCLINICAL STATUS:")
        for obs in perf_obs:
            label = obs["_label"]
            value = obs["_value"]
            if label:
                lines.append(f"  - {label}: {value}" if value else f"  - {label}: (no value)")

    # ── Prior / current treatments ─────────────────────────────────────────────
    if bundle["medications"]:
        lines.append("\nPRIOR / CURRENT TREATMENTS:")
        for med in bundle["medications"]:
            med_cc = med.get("medicationCodeableConcept", {})
            name = med_cc.get("text") or _first_display(med_cc.get("coding", []))
            status = med.get("status", "")
            authored = med.get("authoredOn", "")
            detail = ", ".join(filter(None, [status, authored]))
            lines.append(f"  - {name} ({detail})" if detail else f"  - {name}")

    # ── Allergies ──────────────────────────────────────────────────────────────
    if bundle["allergies"]:
        lines.append("\nALLERGIES:")
        for allergy in bundle["allergies"]:
            code = allergy.get("code", {})
            substance = code.get("text") or _first_display(code.get("coding", []))
            if substance:
                lines.append(f"  - {substance}")

    # ── Procedures ─────────────────────────────────────────────────────────────
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

def _split_criteria(text: str) -> list[dict]:
    """
    Parse a ClinicalTrials.gov eligibility criteria block into a flat list of
    {type, text} dicts.  Handles both bullet (* / - / •) and numbered (1. / 1))
    formats and joins wrapped continuation lines.
    """
    criteria: list[dict] = []
    current_type = "inclusion"
    current_text: str | None = None

    _BULLET = re.compile(r'^[\*\-•]\s+|^\d+[.)]\s+')
    _HEADER = re.compile(r'(inclusion|exclusion)\s+criteria', re.IGNORECASE)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        header_match = _HEADER.search(line)
        if header_match:
            if current_text:
                criteria.append({"type": current_type, "text": current_text.strip()})
                current_text = None
            current_type = "inclusion" if "inclusion" in header_match.group(0).lower() else "exclusion"
            continue

        if _BULLET.match(line):
            if current_text:
                criteria.append({"type": current_type, "text": current_text.strip()})
            current_text = _BULLET.sub("", line).strip()
        elif current_text is not None:
            # continuation of the previous criterion
            current_text += " " + line
        # else: preamble text before first bullet — skip

    if current_text:
        criteria.append({"type": current_type, "text": current_text.strip()})

    return [c for c in criteria if len(c["text"]) > 10]


def _parse_breakdown(raw: str, criteria: list[dict]) -> list[dict]:
    """Merge the LLM's per-criterion verdicts back with the original criterion text."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return [{"error": f"Could not parse breakdown JSON: {raw[:200]}"}]

    items = payload.get("breakdown") or payload.get("criteria") or []
    if not isinstance(items, list):
        return [{"error": "Unexpected breakdown format from model"}]

    result = []
    for item in items:
        n = int(item.get("n", 0))
        idx = n - 1
        criterion_text = criteria[idx]["text"] if 0 <= idx < len(criteria) else item.get("criterion", "")
        ctype = item.get("type") or (criteria[idx]["type"] if 0 <= idx < len(criteria) else "unknown")
        verdict = item.get("verdict", "unknown")
        if verdict not in _VALID_BREAKDOWN_VERDICTS:
            verdict = "unknown"
        result.append({
            "n": n,
            "type": ctype,
            "criterion": criterion_text,
            "verdict": verdict,
            "reason": str(item.get("reason", "")),
        })

    return sorted(result, key=lambda x: x["n"])


def _first_display(coding_list: list[dict]) -> str:
    for coding in coding_list:
        if coding.get("display"):
            return coding["display"]
    return ""



def _parse_response(raw: str) -> dict:
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
