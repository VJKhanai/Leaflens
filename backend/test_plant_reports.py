import unittest
from unittest.mock import patch

from flask import Flask

from backend import plant_reports


class PlantReportTests(unittest.TestCase):
    def setUp(self):
        plant_reports._cache.clear()
        app = Flask(__name__)
        app.register_blueprint(plant_reports.plant_reports_blueprint)
        self.client = app.test_client()

    @patch("backend.plant_reports.get_plant_report")
    def test_endpoint_returns_structured_report(self, get_report):
        get_report.return_value = {"query": "Neem", "status": "partial"}

        response = self.client.get("/plant-info?name=Neem")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"query": "Neem", "status": "partial"})
        get_report.assert_called_once_with("Neem", force_refresh=False)

    @patch("backend.plant_reports.get_plant_report")
    def test_endpoint_validates_names_and_forwards_retry(self, get_report):
        get_report.return_value = {"query": "Neem", "status": "partial"}
        for invalid_name in ("", "n" * 121, "unsafe\nname"):
            response = self.client.get("/plant-info", query_string={"name": invalid_name})
            self.assertEqual(response.status_code, 400)
        get_report.assert_not_called()

        response = self.client.get("/plant-info?name=Neem&refresh=1")

        self.assertEqual(response.status_code, 200)
        get_report.assert_called_once_with("Neem", force_refresh=True)

    @patch("backend.plant_reports._get_source_json")
    def test_builds_report_from_confident_taxonomy_and_literature(self, get_source):
        get_source.side_effect = [
            ({
                "matchType": "EXACT",
                "rank": "SPECIES",
                "confidence": 100,
                "scientificName": "Azadirachta indica A.Juss.",
                "canonicalName": "Azadirachta indica",
                "genus": "Azadirachta",
                "family": "Meliaceae",
                "usageKey": 123,
                "speciesKey": 123,
                "species": "Azadirachta indica",
            }, None),
            ({"results": [{"vernacularName": "Neem", "language": "eng"}]}, None),
            ({
                "results": [{
                    "description": "A flowering evergreen tree.",
                    "source": "Example flora",
                    "citation": "Example flora citation",
                }]
            }, None),
            ({
                "resultList": {
                    "result": [{
                        "id": "PMC12345",
                        "source": "PMC",
                        "title": "A study of Azadirachta indica",
                        "authorString": "Example A",
                        "pubYear": "2024",
                        "firstPublicationDate": "2024-02-01",
                        "abstractText": "Retrieved abstract text.",
                    }]
                }
            }, None),
        ]

        report = plant_reports.get_plant_report("Azadirachta indica")

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["taxonomy"]["family"], "Meliaceae")
        self.assertEqual(report["overview"]["common_names"], ["Neem"])
        self.assertEqual(report["basic_information"]["common_name"], "Neem")
        self.assertEqual(report["basic_information"]["scientific_name"], "Azadirachta indica A.Juss.")
        self.assertEqual(report["basic_information"]["family"], "Meliaceae")
        self.assertEqual(report["basic_information"]["description"], "A flowering evergreen tree.")
        self.assertIsNone(report["basic_information"]["native_region"])
        self.assertEqual(
            report["edibility"]["classification"],
            "Unknown / Insufficient evidence",
        )
        self.assertEqual(
            report["safety_assessment"]["category"],
            "Insufficient safety information",
        )
        self.assertEqual(report["sections"]["benefits"]["status"], "literature_available")
        self.assertEqual(len(report["references"]), 4)
        self.assertEqual(
            report["references"][3]["url"],
            "https://europepmc.org/articles/PMC12345",
        )
        self.assertEqual(report["references"][3]["publication_date"], "2024-02-01")
        self.assertTrue(report["references"][3]["accessed_at"])
        self.assertIn("not been assessed automatically", report["sections"]["benefits"]["message"])

    @patch("backend.plant_reports._get_source_json")
    def test_mint_name_without_scientific_input_does_not_infer_edibility(self, get_source):
        get_source.side_effect = [
            ({
                "matchType": "EXACT",
                "rank": "SPECIES",
                "confidence": 96,
                "scientificName": "Mentha spicata",
                "canonicalName": "Mentha spicata",
                "family": "Lamiaceae",
                "speciesKey": 456,
            }, None),
            ({"results": [{"vernacularName": "Spearmint", "language": "eng"}]}, None),
            ({"results": []}, None),
            ({"resultList": {"result": []}}, None),
        ]

        report = plant_reports.get_plant_report("Mint")

        self.assertEqual(report["basic_information"]["scientific_name"], "Mentha spicata")
        self.assertEqual(report["basic_information"]["common_name"], "Spearmint")
        self.assertEqual(
            report["edibility"]["classification"],
            "Unknown / Insufficient evidence",
        )
        self.assertEqual(report["uses"]["culinary"], [])

    @patch("backend.plant_reports._get_source_json")
    def test_exact_common_name_search_resolves_neem_without_weak_description(self, get_source):
        get_source.side_effect = [
            ({
                "matchType": "NONE",
                "rank": "SPECIES",
                "confidence": 0,
            }, None),
            ({
                "results": [
                    {
                        "rank": "SPECIES",
                        "taxonomicStatus": "ACCEPTED",
                        "canonicalName": "Azadirachta indica",
                        "vernacularNames": [{"vernacularName": "Neem"}],
                    },
                    {
                        "rank": "SPECIES",
                        "taxonomicStatus": "ACCEPTED",
                        "canonicalName": "Azadirachta indica",
                        "vernacularNames": [{"vernacularName": "Neem"}],
                    },
                    {
                        "rank": "SPECIES",
                        "taxonomicStatus": "ACCEPTED",
                        "canonicalName": "Another species",
                        "vernacularNames": [{"vernacularName": "Margosa"}],
                    },
                ]
            }, None),
            ({
                "matchType": "EXACT",
                "rank": "SPECIES",
                "confidence": 99,
                "scientificName": "Azadirachta indica A.Juss.",
                "canonicalName": "Azadirachta indica",
                "family": "Meliaceae",
                "speciesKey": 123,
            }, None),
            ({
                "results": [
                    {"vernacularName": "150 maladies", "language": ""},
                    {"vernacularName": "Indian lilac", "language": "eng"},
                    {"vernacularName": "Neem", "language": "eng"},
                ]
            }, None),
            ({
                "results": [
                    {"description": "Asia", "source": "Example checklist"},
                    {
                        "description": "A lengthy but unattributed text that should not be displayed.",
                        "source": "Example reference",
                    },
                ]
            }, None),
            ({"resultList": {"result": []}}, None),
        ]

        report = plant_reports.get_plant_report("Neem")

        self.assertEqual(report["taxonomy"]["scientific_name"], "Azadirachta indica A.Juss.")
        self.assertEqual(report["basic_information"]["common_name"], "Neem")
        self.assertEqual(
            [entry["name"] for entry in report["basic_information"]["other_names"]],
            ["Indian lilac"],
        )
        self.assertIsNone(report["basic_information"]["description"])
        self.assertNotIn("gbif-description", report["basic_information"]["reference_ids"])
        self.assertEqual(
            report["edibility"]["classification"],
            "Unknown / Insufficient evidence",
        )
        self.assertEqual(get_source.call_count, 6)

    @patch("backend.plant_reports._get_source_json")
    def test_potentially_toxic_plant_is_not_labeled_from_its_name(self, get_source):
        get_source.side_effect = [
            ({
                "matchType": "EXACT",
                "rank": "SPECIES",
                "confidence": 99,
                "scientificName": "Nerium oleander L.",
                "canonicalName": "Nerium oleander",
                "family": "Apocynaceae",
                "speciesKey": 789,
            }, None),
            ({"results": []}, None),
            ({"results": []}, None),
            ({"resultList": {"result": []}}, None),
        ]

        report = plant_reports.get_plant_report("Nerium oleander")

        self.assertEqual(
            report["safety_assessment"]["category"],
            "Insufficient safety information",
        )
        self.assertIsNone(report["disadvantages"]["toxicity_concerns"])
        self.assertEqual(report["edibility"]["classification"], "Unknown / Insufficient evidence")
        self.assertIn("Do not consume", report["safety_assessment"]["ingestion_warning"])

    @patch("backend.plant_reports._get_source_json")
    def test_unknown_plant_returns_all_structured_fields_without_claims(self, get_source):
        get_source.return_value = ({
            "matchType": "NONE",
            "rank": "UNKNOWN",
            "confidence": 0,
        }, None)

        report = plant_reports.get_plant_report("unrecognized leaf label")

        for field in (
            "basic_information",
            "edibility",
            "medicinal_benefits",
            "disadvantages",
            "uses",
            "safety_assessment",
        ):
            self.assertIn(field, report)
        self.assertIsNone(report["basic_information"]["scientific_name"])
        self.assertEqual(report["references"], [])
        self.assertEqual(report["status"], "unavailable")

    @patch("backend.plant_reports._get_source_json")
    def test_external_failures_preserve_unknown_safety_and_explain_unavailable_research(
        self,
        get_source,
    ):
        get_source.side_effect = [
            ({
                "matchType": "EXACT",
                "rank": "SPECIES",
                "confidence": 95,
                "scientificName": "Mentha spicata",
                "family": "Lamiaceae",
                "speciesKey": 456,
            }, None),
            (None, "offline"),
            (None, "offline"),
            (None, "offline"),
        ]

        report = plant_reports.get_plant_report("Mint")

        self.assertEqual(report["safety_assessment"]["category"], "Insufficient safety information")
        self.assertIn("could not be retrieved", report["sections"]["benefits"]["message"])
        self.assertEqual(report["sources"][0]["status"], "partial")
        self.assertIn("message", report["sources"][1])

    @patch("backend.plant_reports._get_source_json")
    def test_rejects_low_confidence_species_match_without_inventing_details(self, get_source):
        get_source.return_value = ({
            "matchType": "FUZZY",
            "rank": "SPECIES",
            "confidence": 54,
            "scientificName": "Example plant",
            "key": 4,
        }, None)

        report = plant_reports.get_plant_report("ambiguous name")

        self.assertEqual(report["status"], "unavailable")
        self.assertIsNone(report["taxonomy"])
        self.assertEqual(report["references"], [])
        self.assertEqual(report["sections"]["edibility"]["status"], "insufficient_information")
        self.assertEqual(get_source.call_count, 2)

    @patch("backend.plant_reports._get_source_json")
    def test_caches_source_failures(self, get_source):
        get_source.return_value = (None, "Source request failed or returned invalid data.")

        first = plant_reports.get_plant_report("Unknown species")
        second = plant_reports.get_plant_report(" Unknown   species ")
        refreshed = plant_reports.get_plant_report("Unknown species", force_refresh=True)

        self.assertIs(first, second)
        self.assertEqual(first["status"], "unavailable")
        self.assertEqual(refreshed["status"], "unavailable")
        self.assertEqual(get_source.call_count, 2)


if __name__ == "__main__":
    unittest.main()
