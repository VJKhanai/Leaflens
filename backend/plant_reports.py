"""Source-attributed plant report retrieval for the LeafLens API."""

import json
import re
import threading
import time
from datetime import datetime, timezone
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from flask import Blueprint, jsonify, request


REQUEST_TIMEOUT_SECONDS = 5
CACHE_TTL_SECONDS = 15 * 60
CACHE_MAX_ENTRIES = 256
USER_AGENT = "LeafLens/1.0 (botanical information lookup)"
plant_reports_blueprint = Blueprint("plant_reports", __name__)

_cache = {}
_cache_lock = threading.Lock()


@plant_reports_blueprint.get("/plant-info")
def plant_info_endpoint():
    name = request.args.get("name", "").strip()
    if not name or len(name) > 120 or any(ord(char) < 32 for char in name):
        return jsonify({"error": "Provide a plant name between 1 and 120 characters."}), 400

    return jsonify(get_plant_report(name, force_refresh=request.args.get("refresh") == "1"))


def _unavailable_sections():
    return {
        key: {
            "status": "insufficient_information",
            "message": message,
        }
        for key, message in {
            "edibility": "Edibility and safe preparation have not been verified by the connected sources. Do not consume this plant based on an AI identification.",
            "nutrition": "No verified, species- and plant-part-specific nutrition data is available from the connected sources.",
            "benefits": "No evidence assessment is available from the connected sources. Research findings are not medical advice.",
            "risks": "Toxicity and interaction information has not been verified by the connected sources.",
            "medicinal_uses": "Traditional or medicinal uses have not been verified by the connected sources.",
            "agricultural_uses": "Agricultural and environmental applications have not been verified by the connected sources.",
            "commercial_uses": "Commercial applications have not been verified by the connected sources.",
            "cultivation": "Species-specific cultivation requirements have not been verified by the connected sources.",
            "safety_summary": "Safety is not established. Do not consume or use medicinally based on an image prediction.",
        }.items()
    }


def _request_json(url):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_source_json(url):
    try:
        return _request_json(url), None
    except (
        HTTPError,
        URLError,
        HTTPException,
        TimeoutError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        return None, "Source request failed or returned invalid data."


def _is_confident_species_match(match):
    return (
        isinstance(match, dict)
        and match.get("matchType") != "NONE"
        and match.get("rank") == "SPECIES"
        and isinstance(match.get("confidence"), (int, float))
        and match["confidence"] >= 90
        and bool(match.get("scientificName"))
        and bool(match.get("speciesKey") or match.get("usageKey") or match.get("key"))
    )


def _find_exact_vernacular_match(name):
    search_url = "https://api.gbif.org/v1/species/search?" + urlencode({
        "q": name,
        "rank": "SPECIES",
        "limit": 20,
    })
    search_data, search_error = _get_source_json(search_url)
    if search_error:
        return None, search_error
    if not isinstance(search_data, dict):
        return None, None

    results = search_data.get("results")
    if not isinstance(results, list):
        return None, None

    query = name.strip().casefold()
    canonical_names = set()
    for candidate in results:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("rank") != "SPECIES" or candidate.get("taxonomicStatus") != "ACCEPTED":
            continue
        canonical_name = candidate.get("canonicalName")
        vernacular_names = candidate.get("vernacularNames")
        if not isinstance(canonical_name, str) or not isinstance(vernacular_names, list):
            continue
        if any(
            isinstance(entry, dict)
            and isinstance(entry.get("vernacularName"), str)
            and entry["vernacularName"].strip().casefold() == query
            for entry in vernacular_names
        ):
            canonical_names.add(canonical_name)

    if len(canonical_names) != 1:
        return None, None

    canonical_name = canonical_names.pop()
    match_url = "https://api.gbif.org/v1/species/match?" + urlencode({"name": canonical_name})
    match, match_error = _get_source_json(match_url)
    if match_error or not _is_confident_species_match(match):
        return None, match_error
    return match, None


def _empty_report(name):
    safety_warning = (
        "The model's leaf-image prediction does not confirm this species. "
        "Do not consume or use medicinally based on an AI identification."
    )
    return {
        "query": name,
        "status": "unavailable",
        "taxonomy": None,
        "basic_information": {
            "common_name": None,
            "scientific_name": None,
            "other_names": [],
            "family": None,
            "description": None,
            "native_region": None,
            "taxonomy_confidence": None,
            "identity_warning": safety_warning,
            "reference_ids": [],
        },
        "edibility": {
            "classification": "Unknown / Insufficient evidence",
            "edible_parts": [],
            "parts_to_avoid": [],
            "preparation": None,
            "toxicity_concerns": None,
            "summary": "Information not verified for this species and plant part.",
            "evidence_status": "insufficient_information",
            "reference_ids": [],
        },
        "medicinal_benefits": {
            "traditional_uses": [],
            "documented_properties": [],
            "parts_traditionally_used": [],
            "evidence_summary": (
                "No medicinal benefit is established by this report. Search results, "
                "when present, are not an assessment of clinical effectiveness."
            ),
            "clinical_evidence_status": "not_assessed",
            "literature_search_results": [],
            "reference_ids": [],
        },
        "disadvantages": {
            "known_side_effects": [],
            "toxicity_concerns": None,
            "allergic_reactions": None,
            "excessive_consumption_risks": None,
            "preparation_risks": None,
            "medication_interactions": None,
            "summary": "Information not verified by the connected sources.",
            "evidence_status": "insufficient_information",
            "reference_ids": [],
        },
        "uses": {
            "traditional_medicinal": [],
            "culinary": [],
            "agricultural_gardening": [],
            "commercial_industrial": [],
            "other_documented": [],
            "evidence_status": "insufficient_information",
            "reference_ids": [],
        },
        "safety_assessment": {
            "category": "Insufficient safety information",
            "summary": "Safety has not been established by the connected sources.",
            "ingestion_warning": safety_warning,
            "reference_ids": [],
        },
        "overview": {
            "status": "insufficient_information",
            "common_names": [],
            "message": "A species-level botanical match was not verified.",
        },
        "sections": _unavailable_sections(),
        "references": [],
        "sources": [
            {"name": "GBIF", "status": "unavailable"},
            {"name": "Europe PMC", "status": "unavailable"},
        ],
    }


def _lookup_uncached(name):
    report = _empty_report(name)
    match_url = "https://api.gbif.org/v1/species/match?" + urlencode({"name": name})
    match, match_error = _get_source_json(match_url)
    gbif_source = report["sources"][0]

    if not match_error and not _is_confident_species_match(match):
        vernacular_match, vernacular_error = _find_exact_vernacular_match(name)
        if vernacular_error:
            match_error = vernacular_error
        elif vernacular_match:
            match = vernacular_match

    if match_error:
        gbif_source["message"] = match_error
    elif not _is_confident_species_match(match):
        gbif_source["status"] = "no_confident_species_match"
        return report
    else:
        scientific_name = match["scientificName"]
        species_key = match.get("speciesKey") or match.get("usageKey") or match.get("key")
        taxonomy_url = "https://www.gbif.org/species/" + quote(str(species_key), safe="")
        accessed_at = datetime.now(timezone.utc).date().isoformat()
        taxonomy = {
            "scientific_name": scientific_name,
            "canonical_name": match.get("canonicalName"),
            "rank": match.get("rank"),
            "family": match.get("family"),
            "genus": match.get("genus"),
            "accepted_name": match.get("species"),
            "confidence": match.get("confidence"),
            "source_url": taxonomy_url,
        }
        report["taxonomy"] = taxonomy
        basic_information = report["basic_information"]
        basic_information.update({
            "scientific_name": scientific_name,
            "family": match.get("family") or None,
            "taxonomy_confidence": match["confidence"],
            "reference_ids": ["gbif-taxonomy"],
        })
        report["overview"] = {
            "status": "partial",
            "common_names": [],
            "message": "GBIF taxonomy is available. Description, habitat, native range, and growing conditions are not provided by this lookup.",
        }
        gbif_source.update({"status": "available", "url": taxonomy_url})
        report["references"].append({
            "id": "gbif-taxonomy",
            "source": "GBIF",
            "title": "GBIF Backbone Taxonomy",
            "author": "Global Biodiversity Information Facility",
            "year": None,
            "publication_date": None,
            "accessed_at": accessed_at,
            "url": taxonomy_url,
            "supports": ["taxonomy"],
        })

        vernacular_url = (
            f"https://api.gbif.org/v1/species/{quote(str(species_key), safe='')}/vernacularNames"
        )
        vernacular_data, vernacular_error = _get_source_json(vernacular_url)
        if vernacular_error or not isinstance(vernacular_data, dict):
            gbif_source.update({
                "status": "partial",
                "message": "Taxonomy was retrieved, but common names were unavailable.",
            })
        else:
            names = []
            seen = set()
            entries = vernacular_data.get("results", [])
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                common_name = entry.get("vernacularName")
                if isinstance(common_name, str) and common_name.strip():
                    normalized = common_name.strip()
                    if normalized.casefold() not in seen:
                        names.append({
                            "name": normalized,
                            "language": entry.get("language") or None,
                        })
                        seen.add(normalized.casefold())
            exact_names = [
                entry for entry in names
                if entry["name"].casefold() == name.strip().casefold()
            ]
            localized_names = [
                entry for entry in names
                if isinstance(entry["language"], str)
                and re.fullmatch(r"[a-zA-Z]{2,3}", entry["language"])
            ]
            preferred_name = (
                exact_names[0]
                if exact_names
                else next(
                    (entry for entry in localized_names if entry["language"].casefold() == "eng"),
                    None,
                )
                or next(iter(localized_names), None)
            )
            displayed_names = (
                [preferred_name]
                + [entry for entry in localized_names if entry != preferred_name]
                if preferred_name
                else []
            )[:10]
            report["overview"]["common_names"] = [
                entry["name"] for entry in displayed_names
            ]
            if preferred_name:
                basic_information["common_name"] = preferred_name["name"]
                basic_information["other_names"] = displayed_names[1:]
                basic_information["reference_ids"].append("gbif-vernacular")
                report["references"].append({
                    "id": "gbif-vernacular",
                    "source": "GBIF",
                    "title": "GBIF vernacular names",
                    "author": "Global Biodiversity Information Facility",
                    "year": None,
                    "publication_date": None,
                    "accessed_at": accessed_at,
                    "url": vernacular_url,
                    "supports": ["common and regional names"],
                })

        description_url = (
            f"https://api.gbif.org/v1/species/{quote(str(species_key), safe='')}/descriptions"
        )
        description_data, description_error = _get_source_json(description_url)
        if description_error or not isinstance(description_data, dict):
            gbif_source.update({
                "status": "partial",
                "message": "Some optional botanical details were unavailable.",
            })
        else:
            description_list = description_data.get("results")
            for entry in description_list if isinstance(description_list, list) else []:
                if not isinstance(entry, dict):
                    continue
                description = entry.get("description")
                attribution = entry.get("citation")
                if (
                    isinstance(description, str)
                    and len(description.strip()) >= 20
                    and isinstance(attribution, str)
                    and attribution.strip()
                ):
                    basic_information["description"] = description.strip()
                    basic_information["reference_ids"].append("gbif-description")
                    report["references"].append({
                        "id": "gbif-description",
                        "source": "GBIF",
                        "title": attribution.strip(),
                        "author": entry.get("source"),
                        "year": None,
                        "publication_date": None,
                        "accessed_at": accessed_at,
                        "url": description_url,
                        "supports": ["botanical description"],
                    })
                    break

        if basic_information["description"]:
            report["overview"]["message"] = (
                "Taxonomy and an attributed description were retrieved from GBIF. "
                "A native range was not verified."
            )
        else:
            report["overview"]["message"] = (
                "Taxonomy is available, but no attributed botanical description or "
                "native range was verified."
            )

        research_query = f'TITLE_ABS:"{scientific_name}"'
        research_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urlencode({
            "query": research_query,
            "format": "json",
            "pageSize": 5,
            "resultType": "core",
        })
        research_data, research_error = _get_source_json(research_url)
        research_source = report["sources"][1]

        if research_error:
            research_source["message"] = research_error
        elif not isinstance(research_data, dict):
            research_source.update({
                "status": "unavailable",
                "message": "Europe PMC returned an invalid response.",
            })
            research_error = True
        else:
            result_list = research_data.get("resultList")
            results = result_list.get("result", []) if isinstance(result_list, dict) else []
            papers = []
            seen_article_ids = set()
            for entry in results if isinstance(results, list) else []:
                if not isinstance(entry, dict):
                    continue
                title = entry.get("title")
                identifier = entry.get("id")
                if not isinstance(title, str) or not isinstance(identifier, str):
                    continue
                if identifier in seen_article_ids:
                    continue
                seen_article_ids.add(identifier)
                article_source = entry.get("source")
                if article_source not in {"MED", "PMC", "PPR"}:
                    article_source = "MED"
                if article_source == "PMC":
                    article_url = f"https://europepmc.org/articles/{quote(identifier, safe='')}"
                else:
                    article_url = (
                        f"https://europepmc.org/article/{article_source}/"
                        f"{quote(identifier, safe='')}"
                    )
                paper = {
                    "title": title,
                    "authors": entry.get("authorString"),
                    "year": entry.get("pubYear"),
                    "publication_date": entry.get("firstPublicationDate") or entry.get("pubYear"),
                    "journal": entry.get("journalTitle"),
                    "url": article_url,
                    "abstract": entry.get("abstractText"),
                }
                papers.append(paper)
                reference_id = f"europe-pmc-{article_source}-{identifier}"
                report["references"].append({
                    "id": reference_id,
                    "source": "Europe PMC",
                    "title": title,
                    "author": entry.get("authorString"),
                    "year": entry.get("pubYear"),
                    "publication_date": entry.get("firstPublicationDate") or entry.get("pubYear"),
                    "accessed_at": accessed_at,
                    "url": article_url,
                    "supports": ["literature search result; relevance and findings not assessed"],
                })

            research_section = report["sections"]["benefits"]
            if papers:
                report["medicinal_benefits"]["literature_search_results"] = papers
                report["medicinal_benefits"]["reference_ids"] = [
                    reference["id"]
                    for reference in report["references"]
                    if reference["source"] == "Europe PMC"
                ]
                report["medicinal_benefits"]["evidence_summary"] = (
                    "These literature search results mention the matched scientific name. "
                    "Their relevance, study quality, outcomes, and clinical significance "
                    "have not been assessed; no treatment benefit is inferred."
                )
                research_section.update({
                    "status": "literature_available",
                    "message": "These search results mention the matched scientific name. Their relevance, study type, outcomes, and clinical significance have not been assessed automatically; do not interpret them as established benefits.",
                    "papers": papers,
                })
                research_source["status"] = "available"
                report["status"] = "partial"
            else:
                research_source["status"] = "no_results"

        if research_error:
            report["sections"]["benefits"] = {
                "status": "insufficient_information",
                "message": "Europe PMC was unavailable; medicinal research information could not be retrieved.",
            }
        elif not report["sections"]["benefits"].get("papers"):
            report["sections"]["benefits"] = {
                "status": "insufficient_information",
                "message": "No relevant literature results were returned for the matched scientific name.",
            }

    if report["status"] != "partial":
        report["status"] = "partial" if report["taxonomy"] else "unavailable"
    return report


def get_plant_report(name, force_refresh=False):
    """Return a cached, source-backed report without inferring unsupported claims."""
    normalized_name = re.sub(r"\s+", " ", name).strip()
    cache_key = normalized_name.casefold()
    now = time.monotonic()

    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and cached[0] > now and not force_refresh:
            return cached[1]
        if force_refresh:
            _cache.pop(cache_key, None)

    report = _lookup_uncached(normalized_name)

    with _cache_lock:
        expired = [key for key, (expires, _) in _cache.items() if expires <= now]
        for key in expired:
            del _cache[key]
        if len(_cache) >= CACHE_MAX_ENTRIES:
            oldest_key = next(iter(_cache))
            del _cache[oldest_key]
        _cache[cache_key] = (now + CACHE_TTL_SECONDS, report)

    return report
