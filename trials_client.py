from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import HTTPError, RequestException, Timeout

CTGOV_V2 = "https://clinicaltrials.gov/api/v2/studies"

# Impersonate Chrome 124 at the TLS level to pass Cloudflare bot detection.
_IMPERSONATE = "chrome124"


class TrialsError(Exception):
    pass


class TrialsClient:
    """Async client for the ClinicalTrials.gov v2 API. Use as async context manager."""

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout
        self._session = AsyncSession(impersonate=_IMPERSONATE)

    async def close(self):
        await self._session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def search(self, conditions: list[str], max_results: int = 10) -> list[dict]:
        """
        Search for RECRUITING trials matching any of the given condition terms.

        Args:
            conditions: List of condition strings (e.g. ["non-small cell lung cancer", "NSCLC"])
            max_results: Cap on returned trials (max 100 per CTG page)

        Returns:
            List of parsed trial dicts.

        Raises:
            TrialsError: on HTTP error or timeout.
        """
        if not conditions:
            return []

        # Quote multi-word terms so CTG treats them as phrases
        quoted = [f'"{t}"' if " " in t else t for t in conditions]
        query_cond = " OR ".join(quoted)

        params = {
            "query.cond": query_cond,
            "filter.overallStatus": "RECRUITING",
            "pageSize": str(min(max_results, 100)),
            "format": "json",
        }

        try:
            resp = await self._session.get(
                CTGOV_V2, params=params, timeout=self._timeout
            )
            resp.raise_for_status()
        except Timeout as exc:
            raise TrialsError("ClinicalTrials.gov request timed out") from exc
        except HTTPError as exc:
            raise TrialsError(
                f"ClinicalTrials.gov returned HTTP {exc.response.status_code}"
            ) from exc
        except RequestException as exc:
            raise TrialsError(f"ClinicalTrials.gov request failed: {exc}") from exc

        studies = resp.json().get("studies", [])
        return [_parse_study(s) for s in studies[:max_results]]


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_study(study: dict) -> dict:
    p = study.get("protocolSection", {})
    id_mod = p.get("identificationModule", {})
    design_mod = p.get("designModule", {})
    desc_mod = p.get("descriptionModule", {})
    elig_mod = p.get("eligibilityModule", {})
    sponsor_mod = p.get("sponsorCollaboratorsModule", {})
    locs_mod = p.get("contactsLocationsModule", {})

    return {
        "nct_id": id_mod.get("nctId", ""),
        "title": id_mod.get("briefTitle", ""),
        "phase": design_mod.get("phases", []),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "locations": _parse_locations(locs_mod.get("locations", [])[:10]),
    }


def _parse_locations(raw: list[dict]) -> list[str]:
    locs = []
    for loc in raw:
        parts = filter(None, [
            loc.get("facility"),
            loc.get("city"),
            loc.get("state"),
            loc.get("country"),
        ])
        label = ", ".join(parts)
        if label:
            locs.append(label)
    return locs


# ── FHIR condition → search term extraction ───────────────────────────────────

def extract_condition_terms(fhir_conditions: list[dict]) -> list[str]:
    """
    Pull the best search terms from a list of FHIR R4 Condition resources.

    Preference order per condition:
      1. SNOMED CT coding display  (most consistent with CTG terminology)
      2. code.text                 (human-readable but may include stage noise)
      3. First available coding display
    """
    seen: set[str] = set()
    terms: list[str] = []

    for cond in fhir_conditions:
        code = cond.get("code", {})
        coding_list = code.get("coding", [])

        candidate = None
        # 1. SNOMED display
        for coding in coding_list:
            if coding.get("system", "").startswith("http://snomed") and coding.get("display"):
                candidate = coding["display"]
                break
        # 2. code.text
        if not candidate:
            candidate = code.get("text")
        # 3. first available display
        if not candidate:
            for coding in coding_list:
                if coding.get("display"):
                    candidate = coding["display"]
                    break

        if candidate and candidate not in seen:
            seen.add(candidate)
            terms.append(candidate)

    return terms
