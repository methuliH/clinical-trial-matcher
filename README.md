# Clinical Trial Matcher

An MCP (Model Context Protocol) server that matches a patient's FHIR record to recruiting clinical trials and explains, criterion by criterion, why the patient is or isn't eligible.

## How it works

1. **Read the patient.** Pulls Patient, Condition, Observation, MedicationRequest, AllergyIntolerance and Procedure resources from a FHIR R4 server.
2. **Find candidate trials.** Derives condition search terms and queries the ClinicalTrials.gov v2 API.
3. **Reason about eligibility.** An LLM (Llama 3.3 70B via the Groq API) compares each trial's eligibility criteria with the patient profile and returns a structured verdict (`met`, `not_met`, `unknown`).

The server advertises the `ai.promptopinion/fhir-context` extension, so SMART-on-FHIR clients can pass FHIR context through HTTP headers (`x-fhir-server-url`, `x-patient-id`, `x-fhir-access-token`) instead of explicit tool arguments.

## MCP tools

| Tool | What it does |
|------|--------------|
| `match_trials` | Finds and ranks candidate trials for a patient |
| `explain_eligibility` | Gives a per-criterion eligibility breakdown for one trial |
| `get_trial_details` | Fetches full trial details by NCT ID |

## Tech stack

Python, FastMCP, httpx, curl_cffi, Groq (OpenAI-compatible API), FHIR R4, ClinicalTrials.gov API v2

## Getting started

```bash
git clone https://github.com/methuliH/clinical-trial-matcher.git
cd clinical-trial-matcher
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key_here" > .env
python main.py                                    # serves on http://localhost:8000 (override with PORT)
```

### Try it with a synthetic patient

```bash
python synthetic_patient.py        # creates an NSCLC test patient on the public HAPI FHIR server
```

It prints a `patient_id` you can pass to `match_trials`.

### Tests

```bash
python test_match_trials.py
python test_two_trials.py
python test_explain.py
```

## Project structure

```
main.py               MCP server and tool definitions
fhir_client.py        Async FHIR R4 client
trials_client.py      ClinicalTrials.gov v2 client
reasoning.py          LLM eligibility reasoning
synthetic_patient.py  Test patient generator
test_*.py             Test scripts
```

## Disclaimer

A research prototype built with synthetic data. It is not medical advice and must not be used for real patient decisions.
