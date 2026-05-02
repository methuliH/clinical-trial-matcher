"""
Creates a synthetic NSCLC patient on a HAPI FHIR R4 server for local testing.

Usage:
    python synthetic_patient.py
    python synthetic_patient.py --fhir-url https://hapi.fhir.org/baseR4

Prints the patient_id and all created resource references so you can pass
patient_id directly to the match_trials MCP tool.
"""
import argparse
import asyncio
import json
import re

import httpx

DEFAULT_FHIR_URL = "https://hapi.fhir.org/baseR4"

_HEADERS = {
    "Accept": "application/fhir+json",
    "Content-Type": "application/fhir+json",
}


async def _post(client: httpx.AsyncClient, url: str, resource: dict) -> dict:
    resp = await client.post(url, json=resource)
    if resp.status_code == 412:
        # HAPI dedup guard: extract the existing resource reference and GET it
        body = resp.json()
        diag = body.get("issue", [{}])[0].get("diagnostics", "")
        match = re.search(r"(\w+/\d+)", diag)
        if match:
            fhir_base = url.rsplit("/", 1)[0]  # strip resource type, keep FHIR base
            get_url = f"{fhir_base}/{match.group(1)}"
            print(f"  (duplicate detected — reusing {match.group(1)})")
            get_resp = await client.get(get_url)
            get_resp.raise_for_status()
            return get_resp.json()
    resp.raise_for_status()
    return resp.json()


async def create_synthetic_patient(fhir_url: str) -> dict:
    async with httpx.AsyncClient(headers=_HEADERS, timeout=60.0) as client:
        base = fhir_url.rstrip("/")

        # ── Patient ───────────────────────────────────────────────────────────
        patient = await _post(client, f"{base}/Patient", {
            "resourceType": "Patient",
            "name": [{"use": "official", "family": "TestNSCLC", "given": ["Alex"]}],
            "gender": "male",
            "birthDate": "1958-03-14",
            "address": [{
                "line": ["123 Clinical Way"],
                "city": "Boston",
                "state": "MA",
                "postalCode": "02101",
                "country": "US",
            }],
        })
        pid = patient["id"]
        pref = f"Patient/{pid}"
        print(f"  Patient:              {pref}")

        # ── Condition: NSCLC Stage IIIB ───────────────────────────────────────
        condition = await _post(client, f"{base}/Condition", {
            "resourceType": "Condition",
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                "code": "active",
            }]},
            "verificationStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                "code": "confirmed",
            }]},
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                "code": "encounter-diagnosis",
            }]}],
            "code": {
                "coding": [
                    {"system": "http://snomed.info/sct", "code": "254637007",
                     "display": "Non-small cell carcinoma of lung"},
                    {"system": "http://hl7.org/fhir/sid/icd-10-cm", "code": "C34.10",
                     "display": "Malignant neoplasm of upper lobe, unspecified bronchus or lung"},
                ],
                "text": "Non-Small Cell Lung Cancer (NSCLC), Stage IIIB",
            },
            "subject": {"reference": pref},
            "onsetDateTime": "2024-09-01",
            "stage": [{"summary": {"coding": [{
                "system": "http://snomed.info/sct",
                "code": "1229954006",
                "display": "Stage IIIB",
            }]}}],
        })
        cref = f"Condition/{condition['id']}"
        print(f"  Condition (NSCLC):    {cref}")

        # ── Observation: ECOG Performance Status 1 ────────────────────────────
        ecog = await _post(client, f"{base}/Observation", {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "survey",
            }]}],
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "89243-0",
                             "display": "ECOG Performance Status"}],
                "text": "ECOG Performance Status",
            },
            "subject": {"reference": pref},
            "effectiveDateTime": "2025-01-15",
            "valueCodeableConcept": {
                "coding": [{"system": "http://loinc.org", "code": "LA9623-5",
                             "display": "ECOG 1 – Restricted in strenuous activity; ambulatory"}],
                "text": "ECOG 1",
            },
        })
        print(f"  Observation (ECOG):   Observation/{ecog['id']}")

        # ── Observation: PD-L1 TPS 45 % ───────────────────────────────────────
        pdl1 = await _post(client, f"{base}/Observation", {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory",
            }]}],
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "85319-2",
                             "display": "PD-L1 by immunohistochemistry"}],
                "text": "PD-L1 Expression (TPS)",
            },
            "subject": {"reference": pref},
            "effectiveDateTime": "2024-10-05",
            "valueQuantity": {
                "value": 45,
                "unit": "%",
                "system": "http://unitsofmeasure.org",
                "code": "%",
            },
        })
        print(f"  Observation (PD-L1):  Observation/{pdl1['id']}")

        # ── Observation: EGFR mutation – negative ─────────────────────────────
        egfr = await _post(client, f"{base}/Observation", {
            "resourceType": "Observation",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                "code": "laboratory",
            }]}],
            "code": {
                "coding": [{"system": "http://loinc.org", "code": "81311-2",
                             "display": "EGFR gene mutation analysis"}],
                "text": "EGFR Mutation Status",
            },
            "subject": {"reference": pref},
            "effectiveDateTime": "2024-10-05",
            "valueCodeableConcept": {
                "coding": [{"system": "http://snomed.info/sct", "code": "260385009",
                             "display": "Negative"}],
                "text": "EGFR Mutation Negative",
            },
        })
        print(f"  Observation (EGFR):   Observation/{egfr['id']}")

        # ── MedicationRequest: Carboplatin (completed prior line) ─────────────
        med = await _post(client, f"{base}/MedicationRequest", {
            "resourceType": "MedicationRequest",
            "status": "completed",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                             "code": "40048", "display": "Carboplatin"}],
                "text": "Carboplatin 400 mg IV",
            },
            "subject": {"reference": pref},
            "authoredOn": "2024-11-01",
            "reasonReference": [{"reference": cref}],
        })
        print(f"  MedicationRequest:    MedicationRequest/{med['id']}")

        # ── AllergyIntolerance: Penicillin ────────────────────────────────────
        allergy = await _post(client, f"{base}/AllergyIntolerance", {
            "resourceType": "AllergyIntolerance",
            "clinicalStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                "code": "active",
            }]},
            "verificationStatus": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification",
                "code": "confirmed",
            }]},
            "type": "allergy",
            "category": ["medication"],
            "criticality": "low",
            "code": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                             "code": "7980", "display": "Penicillin"}],
                "text": "Penicillin",
            },
            "patient": {"reference": pref},
            "reaction": [{"manifestation": [{"coding": [{
                "system": "http://snomed.info/sct",
                "code": "247472004",
                "display": "Hives",
            }]}]}],
        })
        print(f"  AllergyIntolerance:   AllergyIntolerance/{allergy['id']}")

        # ── Procedure: CT-guided biopsy ───────────────────────────────────────
        procedure = await _post(client, f"{base}/Procedure", {
            "resourceType": "Procedure",
            "status": "completed",
            "code": {
                "coding": [{"system": "http://snomed.info/sct", "code": "432231000",
                             "display": "CT-guided biopsy of thorax"}],
                "text": "CT-guided lung biopsy",
            },
            "subject": {"reference": pref},
            "performedDateTime": "2024-09-15",
            "reasonReference": [{"reference": cref}],
        })
        print(f"  Procedure:            Procedure/{procedure['id']}")

        return {
            "patient_id": pid,
            "fhir_base_url": fhir_url,
            "resources": {
                "patient": pref,
                "condition": cref,
                "ecog_observation": f"Observation/{ecog['id']}",
                "pdl1_observation": f"Observation/{pdl1['id']}",
                "egfr_observation": f"Observation/{egfr['id']}",
                "medication_request": f"MedicationRequest/{med['id']}",
                "allergy_intolerance": f"AllergyIntolerance/{allergy['id']}",
                "procedure": f"Procedure/{procedure['id']}",
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create synthetic NSCLC patient on HAPI FHIR R4")
    parser.add_argument("--fhir-url", default=DEFAULT_FHIR_URL,
                        help=f"FHIR base URL (default: {DEFAULT_FHIR_URL})")
    args = parser.parse_args()

    print(f"Creating synthetic NSCLC patient on {args.fhir_url} …\n")
    result = asyncio.run(create_synthetic_patient(args.fhir_url))

    print("\n--- Result ---")
    print(json.dumps(result, indent=2))
    print(f"\nTest the MCP tool:\n"
          f"  fhir_base_url = \"{result['fhir_base_url']}\"\n"
          f"  patient_id    = \"{result['patient_id']}\"")
