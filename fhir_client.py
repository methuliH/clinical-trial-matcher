import asyncio
from typing import Optional

import httpx


class FHIRError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"FHIR {status_code}: {message}")


class FHIRClient:
    """Async FHIR R4 client. Use as an async context manager or call close() manually."""

    def __init__(self, base_url: str, access_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        headers = {
            "Accept": "application/fhir+json",
            "Content-Type": "application/fhir+json",
        }
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"
        self._client = httpx.AsyncClient(headers=headers, timeout=30.0)

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _check(self, response: httpx.Response) -> None:
        if response.is_error:
            try:
                body = response.json()
                if body.get("resourceType") == "OperationOutcome":
                    issues = body.get("issue", [])
                    msg = "; ".join(
                        i.get("diagnostics") or i.get("details", {}).get("text", "unknown")
                        for i in issues
                    )
                else:
                    msg = str(body)[:300]
            except Exception:
                msg = response.text[:300]
            raise FHIRError(response.status_code, msg)

    async def _search(self, resource_type: str, params: dict) -> list[dict]:
        resp = await self._client.get(
            f"{self.base_url}/{resource_type}",
            params={**params, "_count": "100"},
        )
        self._check(resp)
        return [entry["resource"] for entry in resp.json().get("entry", [])]

    async def get_patient(self, patient_id: str) -> dict:
        resp = await self._client.get(f"{self.base_url}/Patient/{patient_id}")
        self._check(resp)
        return resp.json()

    async def get_conditions(self, patient_id: str) -> list[dict]:
        return await self._search("Condition", {"patient": patient_id})

    async def get_observations(self, patient_id: str) -> list[dict]:
        return await self._search("Observation", {"patient": patient_id, "_sort": "-date"})

    async def get_medication_requests(self, patient_id: str) -> list[dict]:
        return await self._search("MedicationRequest", {"patient": patient_id})

    async def get_allergy_intolerances(self, patient_id: str) -> list[dict]:
        return await self._search("AllergyIntolerance", {"patient": patient_id})

    async def get_procedures(self, patient_id: str) -> list[dict]:
        return await self._search("Procedure", {"patient": patient_id})

    async def get_patient_bundle(self, patient_id: str) -> dict:
        """Fetches all six resource types concurrently and returns a normalized bundle."""
        (
            patient,
            conditions,
            observations,
            medications,
            allergies,
            procedures,
        ) = await asyncio.gather(
            self.get_patient(patient_id),
            self.get_conditions(patient_id),
            self.get_observations(patient_id),
            self.get_medication_requests(patient_id),
            self.get_allergy_intolerances(patient_id),
            self.get_procedures(patient_id),
        )
        return {
            "patient": patient,
            "conditions": conditions,
            "observations": observations,
            "medications": medications,
            "allergies": allergies,
            "procedures": procedures,
        }
