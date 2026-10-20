import unittest
from unittest.mock import patch

from src.external_research import fetch_external_source


def valid_result():
    return {
        "title": "UU Perlindungan Data Pribadi",
        "url": "https://peraturan.bpk.go.id/Details/123",
        "snippet": "Ringkasan regulasi.",
        "source_name": "JDIH BPK",
        "domain": "peraturan.bpk.go.id",
        "source_tier": 1,
    }


class TestExternalResearchV63(unittest.TestCase):

    @patch("src.external_research._fetch")
    def test_fetch_valid_source(self, mock_fetch):
        mock_fetch.return_value = """
        <html>
          <head><title>Regulasi</title></head>
          <body>
            <h1>UU Perlindungan Data Pribadi</h1>
            <p>Ketentuan mengenai pelindungan data pribadi.</p>
          </body>
        </html>
        """

        result = fetch_external_source(valid_result())

        self.assertEqual(result["status"], "ok")
        self.assertIn("Ketentuan mengenai", result["evidence"])
        self.assertEqual(result["source"]["source_name"], "JDIH BPK")
        self.assertGreater(result["evidence_chars"], 0)

    @patch("src.external_research._fetch")
    def test_html_is_cleaned(self, mock_fetch):
        mock_fetch.return_value = """
        <html>
          <script>alert("ignore");</script>
          <style>.x { display:none; }</style>
          <body>Pasal 1 tentang data pribadi.</body>
        </html>
        """

        result = fetch_external_source(valid_result())

        self.assertEqual(result["status"], "ok")
        self.assertIn("Pasal 1", result["evidence"])
        self.assertNotIn("<script", result["evidence"])
        self.assertNotIn("display:none", result["evidence"])

    @patch("src.external_research._fetch")
    def test_evidence_is_bounded(self, mock_fetch):
        mock_fetch.return_value = "<html><body>" + \
            ("A" * 5000) + "</body></html>"

        result = fetch_external_source(valid_result(), max_chars=1000)

        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(result["evidence_chars"], 1000)

    def test_invalid_source_is_rejected_before_fetch(self):
        invalid = valid_result()
        invalid["url"] = "https://example.com/regulation"

        with patch("src.external_research._fetch") as mock_fetch:
            result = fetch_external_source(invalid)

        self.assertEqual(result["status"], "invalid_source")
        mock_fetch.assert_not_called()

    @patch("src.external_research._fetch")
    def test_fetch_failure_is_controlled(self, mock_fetch):
        from urllib.error import URLError
        mock_fetch.side_effect = URLError("network failure")

        result = fetch_external_source(valid_result())

        self.assertEqual(result["status"], "fetch_failed")
        self.assertEqual(result["evidence"], "")

    @patch("src.external_research._fetch")
    def test_empty_source_is_controlled(self, mock_fetch):
        mock_fetch.return_value = "<html><body></body></html>"

        result = fetch_external_source(valid_result())

        self.assertEqual(result["status"], "empty_source")
        self.assertEqual(result["evidence"], "")


if __name__ == "__main__":
    unittest.main()
