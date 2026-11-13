import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from backend.app import app as existing_app


def vit_result():
    all_predictions = [
        {
            "class_index": index,
            "class_name": f"Plant {index}",
            "confidence_percent": round(100 / (index + 1), 2),
        }
        for index in range(80)
    ]
    return {
        "predictions": all_predictions[:5],
        "all_predictions": all_predictions,
        "class_count": 80,
    }


class ExistingPredictionProxyTests(unittest.TestCase):
    def setUp(self):
        self.client = existing_app.test_client()

    @patch("backend.app.urlopen")
    def test_existing_predict_route_proxies_and_preserves_full_vit_output(self, urlopen_mock):
        response_body = json.dumps(vit_result()).encode("utf-8")
        upstream_response = urlopen_mock.return_value.__enter__.return_value
        upstream_response.read.return_value = response_body

        response = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(b"sample image"), "leaf.png", "image/png")},
            content_type="multipart/form-data",
        )

        payload = response.get_json()
        upstream_request = urlopen_mock.call_args.args[0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction"], "Plant 0")
        self.assertEqual(payload["class_count"], 80)
        self.assertEqual(len(payload["predictions"]), 5)
        self.assertEqual(len(payload["all_predictions"]), 80)
        self.assertIn("include_all=true", upstream_request.full_url)
        self.assertIn(b"sample image", upstream_request.data)

    @patch("backend.app.urlopen", side_effect=URLError("offline"))
    def test_reports_vi_t_service_unavailability_without_internal_details(self, _urlopen_mock):
        response = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(b"sample image"), "leaf.png", "image/png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_json(),
            {"error": "The plant prediction service is unavailable. Please try again."},
        )

    @patch("backend.app.urlopen")
    def test_rejects_unsupported_upload_without_calling_vit(self, urlopen_mock):
        response = self.client.post(
            "/predict",
            data={"file": (io.BytesIO(b"not an image"), "leaf.webp", "image/webp")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        urlopen_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
